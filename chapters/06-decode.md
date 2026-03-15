# 第 6 章 Decode 过程

> "One token at a time, but never one request at a time." —— Decode 阶段每步只生成一个 token，但通过 batching 让多个请求共享同一次 GPU 计算。

Prefill 阶段为序列铺设好了 KV cache，decode 阶段则在此基础上逐 token 生成。每一步 decode 只需要计算一个新 token 的 query 与整条序列的 KV 做 attention，因此单步计算量很小，但需要反复执行直到序列完成。Mini-SGLang 通过 CUDA graph、高效的 KV cache 读取和灵活的 sampling 策略，将这一过程优化到极致。

---

## 6.1 单步 Token 生成循环

Decode 的核心是一个不断重复的循环：读取 KV cache -> 模型前向计算 -> sampling -> 追加 token -> 检查是否结束。在 Mini-SGLang 中，这个循环并非显式的 `while` 循环，而是由 scheduler 的主循环驱动——每一轮调度都可能产生一个 decode batch。

`DecodeManager` 维护着正在生成的请求集合：

```python
# 文件: python/minisgl/scheduler/decode.py
@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0
```

每次调度时，只要 `running_reqs` 非空，就将所有正在 decode 的请求打包成一个 batch。这种设计比逐条请求处理高效得多——GPU 上一次 batch forward 的开销与单条请求相差不大，但吞吐量成倍增长。

---

## 6.2 KV Cache 读取与 FlashInfer

Decode 阶段使用 FlashInfer（而非 prefill 阶段的 FlashAttention）作为 attention backend。原因在于 decode 的 query 长度始终为 1，FlashInfer 针对这种 "单 query 对长 key-value" 的模式做了专门优化。

在每步 decode 中：
- 新 token 的 K、V 通过 `store_kv` 写入 cache
- Attention 计算读取该请求的全部历史 KV

由于 decode 阶段每个请求的 `extend_len = 1`（`complete_one` 将 `cached_len` 推进到 `device_len - 1`，只有最新的一个 token 未被 cache），attention 的 query 维度极小，瓶颈在于 KV cache 的内存读取而非计算。FlashInfer 的 paged attention 实现高效地从分散的 page 中读取 KV 数据。

---

## 6.3 CUDA Graph 优化

Decode 阶段的一大特点是**计算模式高度固定**：每个请求贡献 1 个 query token，batch 中每个请求的计算量相同。Mini-SGLang 利用这一特性，通过 CUDA graph 消除 kernel launch 的开销。

### GraphRunner 的预捕获

`GraphRunner` 在引擎初始化时，针对一系列预定义的 batch size 捕获 CUDA graph：

```python
# 文件: python/minisgl/engine/graph.py
def _determine_cuda_graph_bs(...) -> List[int]:
    ...
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
```

典型的 batch size 列表为 `[1, 2, 4, 8, 16, 24, ..., 160]`（或 H200 上到 256）。捕获过程从大到小遍历，使用同一个 graph pool 共享内存：

```python
# 文件: python/minisgl/engine/graph.py
def _capture_graphs(self, max_seq_len, vocab_size, model):
    ...
    for bs in pbar:
        graph = torch.cuda.CUDAGraph()
        batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
        batch.padded_reqs = batch.reqs
        self.attn_backend.prepare_for_capture(batch)
        self.buffer.set_batch(batch)
        with get_global_ctx().forward_batch(batch):
            self.buffer.logits[:bs] = model.forward()  # warmup
            with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                self.buffer.logits[:bs] = model.forward()  # capture
        if pool is None:
            pool = graph.pool()
        self.graph_map[bs] = graph
```

### Batch Padding

由于 CUDA graph 要求固定的 tensor shape，实际 batch size 必须与预捕获的 size 对齐。`pad_batch` 方法用 `dummy_req` 填充到最近的预捕获 size：

```python
# 文件: python/minisgl/engine/graph.py
def pad_batch(self, batch: Batch) -> None:
    padded_size = (
        next(bs for bs in self.graph_bs_list if bs >= batch.size)
        if self.can_use_cuda_graph(batch)
        else batch.size
    )
    batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
```

例如，13 个请求的 batch 会被 padding 到 16。多出的 3 个 dummy request 参与计算但结果被丢弃。这看似浪费，但相比 kernel launch 的节省，是值得的。

### Replay

实际推理时，`replay` 方法将当前 batch 的数据拷入预分配的 buffer，然后重放 graph：

```python
# 文件: python/minisgl/engine/graph.py
def replay(self, batch: Batch) -> torch.Tensor:
    self.buffer.copy_from(batch)
    g = self.graph_map[batch.padded_size]
    self.attn_backend.prepare_for_replay(batch)
    g.replay()
    return self.buffer.logits[: batch.size]
```

注意返回时只取 `batch.size` 个 logits，padding 部分被裁掉。

---

## 6.4 Logits Sampling

模型输出 logits 后，`Sampler` 负责从中选出下一个 token。Mini-SGLang 支持 greedy decoding 和基于概率的 sampling：

```python
# 文件: python/minisgl/engine/sample.py
def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    if args.temperatures is None:  # greedy sampling
        return torch.argmax(logits, dim=-1)
    return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
```

**Greedy decoding**（`temperature = 0`）直接取 argmax，零随机性。当 batch 内所有请求都使用 greedy 时，`prepare` 方法返回 `temperatures=None`，避免不必要的 tensor 分配。

**概率 sampling** 则通过 flashinfer 的 sampling 库实现，支持 top-k、top-p 及其组合：

```python
# 文件: python/minisgl/engine/sample.py
def sample_impl(logits, temperatures, top_k, top_p):
    probs = sampling.softmax(logits, temperatures, ...)
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)
    if top_p is None:
        return sampling.top_k_sampling_from_probs(probs, top_k)
    if top_k is None:
        return sampling.top_p_sampling_from_probs(probs, top_p)
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)
```

`prepare` 方法中有一个巧妙的优化：只有当 batch 中存在非 greedy 请求时才创建 temperature tensor，只有当存在有效的 top-k 或 top-p 值时才创建对应 tensor，减少了不必要的 GPU 内存分配。

---

## 6.5 Token 追加与完成判定

Sampling 结果经异步拷贝回 CPU 后，`_process_last_data` 为每个请求追加新 token 并检查是否完成：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _process_last_data(self, last_data):
    ...
    for i, req in enumerate(batch.reqs):
        if isinstance(req, ChunkedReq):
            continue
        next_token = next_tokens_cpu[i]
        req.append_host(next_token.unsqueeze(0))
        finished = not req.can_decode
        if not req.sampling_params.ignore_eos:
            finished |= next_token == self.eos_token_id
        ...
        if finished:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
```

`append_host` 将新 token 拼接到 `input_ids`（CPU 上），保持 host 端的完整序列记录。完成判定有两个条件：

1. **长度耗尽**：`req.can_decode` 返回 `False`，即 `remain_len <= 0`，已达到 `max_device_len`。
2. **遇到 EOS token**：除非 `ignore_eos` 被设置，否则生成 EOS 即终止。

完成的请求从 `DecodeManager.running_reqs` 中移除，KV cache 资源被释放。未完成的请求继续留在 `running_reqs` 中，等待下一轮 decode。

`complete_one` 方法在 engine 的 `forward_batch` 中被调用，将 `cached_len` 推进到当前 `device_len`，并将 `device_len` 加 1，为下一步 decode 预留位置。这保证了 decode 阶段每步的 `extend_len` 始终为 1。

---

## 本章小结

1. **Decode 阶段每步只计算 1 个 query token**，但通过 batching 将多个请求合并执行，极大提升 GPU 利用率。
2. **FlashInfer** 专为 decode 场景（单 query、长 KV）优化，替代 prefill 阶段使用的 FlashAttention。
3. **CUDA graph** 预捕获一系列固定 batch size 的计算图，decode 时通过 replay 消除 kernel launch 开销，batch 通过 dummy request padding 对齐到预捕获 size。
4. **Sampling** 支持 greedy（argmax）和概率采样（top-k、top-p），通过条件化的 tensor 分配避免不必要的开销。
5. **`complete_one`** 每步推进 `cached_len` 和 `device_len`，保证 decode 的 `extend_len` 始终为 1。
6. **完成判定** 基于长度限制和 EOS token 两个条件，完成的请求立即释放资源。
