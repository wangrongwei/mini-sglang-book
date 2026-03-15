# 第 5 章 Prefill 过程

> "The first token is always the hardest." —— 在自回归生成中，prefill 阶段一次性处理完所有输入 token，为后续逐 token 生成铺设好 KV cache 的基石。

大语言模型的推理分为两个截然不同的阶段：**prefill**（预填充）和 **decode**（解码）。Prefill 阶段接收用户的完整输入序列，经过 tokenization、KV cache 分配、attention 计算，最终产出第一个生成 token。这一阶段的计算量远大于单步 decode，因为它需要一次性处理整条输入序列。Mini-SGLang 用极简的代码结构清晰地展现了这一过程。

---

## 5.1 输入 Tokenization 与请求构造

当用户请求到达时，scheduler 将其封装为一个 `Req` 对象。`Req` 中最核心的字段是 `input_ids`——一个 CPU 上的 tensor，存储 tokenize 后的 token ID 序列：

```python
# 文件: python/minisgl/core.py
@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```

注意 `cached_len` 和 `device_len` 这两个字段：`device_len` 表示当前序列在设备上的逻辑长度（初始为输入长度），`cached_len` 表示已经写入 KV cache 的 token 数量。两者的差值就是本次 prefill 需要实际计算的 token 数。

---

## 5.2 extend_len：理解增量计算

Mini-SGLang 引入了 `extend_len` 的概念，这是理解 prefill 过程的关键：

```python
# 文件: python/minisgl/core.py
@property
def extend_len(self) -> int:
    return self.device_len - self.cached_len
```

对于一个全新的请求，`cached_len = 0`，`extend_len` 等于整个输入长度。但当 RadixPrefixCache 命中了部分前缀时，`cached_len` 会大于 0，`extend_len` 就只包含未命中的那部分 token。这是一个精妙的简化：无论是否有 prefix cache 命中，后续流程都统一处理 `extend_len` 个 token，无需分支逻辑。

---

## 5.3 Prefill Batch 的准备

`PrefillManager` 负责从等待队列中挑选请求，组装成一个 prefill batch：

```python
# 文件: python/minisgl/scheduler/prefill.py
class PrefillAdder:
    token_budget: int
    reserved_size: int
    cache_manager: CacheManager
    table_manager: TableManager

    def try_add_one(self, req, ...) -> bool:
        # 尝试分配 cache 资源
        # 检查 token_budget 是否足够
        # 如果 token 超出预算，创建 ChunkedReq（分块请求）
        ...
```

`PrefillAdder` 维护一个 `token_budget`，逐个尝试添加请求。当一个请求的 `extend_len` 超出剩余预算时，Mini-SGLang 会创建 `ChunkedReq`——只处理部分 token，剩余部分留到下一个 batch。这种 chunked prefill 机制保证了单次 batch 不会因为一条超长输入而占用过多 GPU 资源。

调度器的 `_schedule_next_batch` 优先安排 prefill：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

---

## 5.4 Positions Tensor 的构建

Prefill 需要为每个 token 提供正确的位置编码。`_make_positions` 函数为 batch 中每个请求生成从 `cached_len` 到 `device_len` 的位置索引：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len, req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)
```

如果请求的 `cached_len = 100`，`device_len = 200`，那么 positions 就是 `[100, 101, ..., 199]`——只为需要计算的 token 生成位置，而非整条序列。这里使用 pinned memory（`pin_memory=True`）加速 host-to-device 传输，是一个常见的工程优化。

---

## 5.5 Attention 计算：FlashAttention-3

Prefill 阶段的 attention 使用 FlashAttention（版本 3 或 4，取决于 GPU 架构）。其 `forward` 方法先将 K、V 写入 cache，再调用 FlashAttention kernel：

```python
# 文件: python/minisgl/attention/fa.py
def forward(self, q, k, v, layer_id, batch):
    metadata = batch.attn_metadata
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        cu_seqlens_q=metadata.cu_seqlens_q,
        cu_seqlens_k=metadata.cu_seqlens_k,
        max_seqlen_q=metadata.max_seqlen_q,
        softmax_scale=self.scale,
        version=self.version,
    )
```

`prepare_metadata` 方法为 FlashAttention 构建所需的 cumulative sequence lengths：

```python
# 文件: python/minisgl/attention/fa.py
seqlens_q = [req.extend_len for req in reqs]   # query 长度 = extend_len
seqlens_k = [req.device_len for req in reqs]    # key 长度 = 完整序列长度
```

这里 `seqlens_q` 使用 `extend_len`（只计算新 token 的 query），而 `seqlens_k` 使用 `device_len`（attention 需要看到包括 cache 命中在内的完整 key 序列）。这就是为什么 prefix cache 能减少计算量：query 变短了，但 key 还是完整的。

---

## 5.6 KV Cache 写入与首 Token 采样

`forward_batch` 在模型前向计算完成后，调用 `complete_one` 更新每个请求的状态，然后进行 sampling：

```python
# 文件: python/minisgl/engine/engine.py
def forward_batch(self, batch, args):
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model.forward()

    for req in batch.reqs:
        req.complete_one()  # cached_len = device_len; device_len += 1

    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args)
    ...
```

`complete_one` 做了两件事：将 `cached_len` 推进到 `device_len`（表示这些 token 的 KV 已经被写入 cache），然后将 `device_len` 加 1（为即将到来的 decode 步预留位置）。Prefill 结束后，请求自然进入 decode 阶段。

注意 prefill batch 通常不使用 CUDA graph（因为输入长度可变），而是直接调用 `model.forward()`。

---

## 本章小结

1. **Prefill 是一次性处理完整输入序列的阶段**，其计算量与输入长度成正比，远大于单步 decode。
2. **`extend_len = device_len - cached_len`** 是核心概念，统一了有无 prefix cache 命中的处理逻辑。
3. **Positions tensor** 只为 `extend_len` 范围内的 token 生成，结合 pinned memory 实现高效传输。
4. **FlashAttention** 的 `seqlens_q` 使用 `extend_len`，`seqlens_k` 使用 `device_len`，使 prefix cache 命中能真正减少 query 侧的计算量。
5. **`complete_one` 方法**在前向计算后更新请求状态，将 `cached_len` 对齐到 `device_len`，并为 decode 阶段预留空间。
6. **Chunked prefill** 通过 `ChunkedReq` 支持超长输入的分批处理，避免单条请求耗尽 GPU 资源。
