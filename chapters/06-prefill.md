# 第 6 章 Prefill 过程

> "The first token is always the hardest." —— 在自回归生成中，prefill 阶段一次性处理完所有输入 token，为后续逐 token 生成铺设好 KV cache 的基石。

大语言模型的推理分为两个截然不同的阶段：**prefill**（预填充）和 **decode**（解码）。Prefill 阶段接收用户的完整输入序列，经过 tokenization、KV cache 分配、attention 计算，最终产出第一个生成 token。这一阶段的计算量远大于单步 decode，因为它需要一次性处理整条输入序列。Mini-SGLang 用极简的代码结构清晰地展现了这一过程。

---

## 6.1 输入 Tokenization 与请求构造
### 6.1.1 prefill manager
前面提到，在 Scheduler._schedule_next_batch()函数中，会优先处理 prefill 阶段的请求。
```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        return Batch(reqs=reqs, phase="prefill")
```
这里包含了 LLM 推理引擎中一个非常高级的特性——Chunked Prefill（分块预填充）。以下是具体的流程拆解：预算与资源评估、遍历请求并加入 Batch、更新队列与返回 Batch。

#### 6.1.1.1 预算与资源评估

```python
if len(self.pending_list) == 0:
    return None

adder = PrefillAdder(
    token_budget=prefill_budget,
    reserved_size=self.decode_manager.inflight_tokens,
    cache_manager=self.cache_manager,
    table_manager=self.table_manager,
)
```
- 检查空队列：如果没有新请求排队，直接返回 None。
- 初始化 PrefillAdder 评估器：这是一个专门用来“算账”的辅助对象。
    - token_budget：算力预算（比如限制这次最多只能算 4096 个 Token，防止 GPU 显存 OOM 或单次耗时过长导致其他对话卡顿）。
    - reserved_size：必须为正在进行 Decode（吐字）的老请求预留显存空间，不能把资源全部分配给新请求。
    - cache_manager / table_manager：用于检查 KV Cache 物理显存页是否还够用。

为一笔全新的请求尝试分配底层资源（包括 KV Cache 块和页表/Token槽位），并尽可能复用系统里已有的 Prefix Cache。 如果系统当前显存或槽位资源不足，则分配失败返回 None。
这个函数主要在处理一个尚未进行过计算的 PendingReq 时被调用。
```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
            --> try_add_one
                --> __try_allocate_one

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 判断是否有空闲的序列槽位
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # 将请求交给 cache_manager 去匹配。系统会在底层（通常是一棵 Radix Tree / Prefix Tree）
        # 查找该请求的 input_ids 是否有部分已经被计算过并缓存在显存中（比如 System Prompt 或者历史对话）。
        # 
        # handle: 匹配到的缓存句柄（包含了匹配到了哪些物理页）。
        # cached_len: 成功匹配上的 Token 长度。
        handle = self.cache_manager.match_req(req).cuda_handle
        cached_len = handle.cached_len
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len
        estimated_len = extend_len + req.output_len

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)

        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx
```
上面重要的主要有两点：
- 资源预估：计算这个请求实际还需要计算的前缀长度（extend_len），并加上用户请求的最大生成长度（req.output_len），预估出这个请求未来最多会额外消耗多少显存资源（estimated_len）。预估后，进行可用性检查（Double Check）：
    - 判断：需要的显存 + 已经预留给其他请求的显存 > 系统当前剩余的显存。如果超出，则放弃调度。
    - 加锁 (Lock)：一旦确认初步空间足够，调用 lock(handle) 将刚才匹配到的历史 Cache 块锁定，防止在本次调度过程中这些块被 LRU 策略驱逐（Evict）掉。
    - 二次检查：加锁后再次确认可用资源。如果此时发现剩余空间不足（可能加锁导致了一些内部计算或可用空间变动），则安全地解锁并返回 None。
- 真正分配槽位并装载前缀状态
    - 调用 allocate() 真正在 table_manager 里为这个请求分配一个索引（table_idx）。
    - 组装 Cache 状态：如果刚才匹配到了历史缓存，就把命中部分的 Token IDs 和 物理块的索引（get_matched_indices()）以异步非阻塞（non_blocking=True）的方式直接拷贝到新分配的 table_idx 对应的空间里。
    - 这样一来，底层的 Attention 算子在计算时，就能直接从页表里找到前面已经算好的 KV 缓存了。

#### 6.1.1.2 遍历请求并加入 Batch

回到请求处理的视角。
```python
reqs: List[Req] = []
chunked_list: List[PendingReq] = []
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        pending_req.chunked_req = None
        if isinstance(req, ChunkedReq):
            pending_req.chunked_req = req
            chunked_list.append(pending_req)
        reqs.append(req)
    else:
        break  # 预算耗尽，停止添加
```
循环遍历排队的新请求，利用 adder.try_add_one() 尝试将其加入当前的批次中。
```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
            --> try_add_one

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        if self.token_budget <= 0:
            return None

        if chunked_req := pending_req.chunked_req: # 属于未读完的半截请求
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req): # 完全新的请求
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None
```
因为有分块预填充（Chunked Prefill）机制，如果上一次调度时用户的长文本没算完（比如被截断了一半），它的状态会被保存在 pending_req.chunked_req 中。这里优先判断它是不是一个“历史遗留”的半截请求：如果是，不需要重新去申请内存或匹配 Radix Tree 前缀，直接拿出它之前算了一半的 cache_handle（缓存句柄）、table_idx（物理页表位置）和 cached_len（已经算到了第几个字），传给 _add_one_req 继续往下算。

```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
            --> try_add_one
                --> __try_allocate_one
                --> _add_one_req

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        remain_len = pending_req.input_len - cached_len
        chunk_size = min(self.token_budget, remain_len)
        is_chunked = chunk_size < remain_len
        CLS = ChunkedReq if is_chunked else Req
        self.token_budget -= chunk_size
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the 
        # 这里执行了一次高效的数据传输。它找到设备端（如 GPU）分配给该请求的 Token Pool 的具体位置（device_ids）。
        # 提取出本次 Chunk 需要处理的 CPU 端的 Token IDs (pending_req.input_ids[_slice])。
        # 使用 .pin_memory()（锁页内存）和 non_blocking=True 进行异步的、非阻塞的主机到设备（Host-to-Device）
        # 内存拷贝。这可以隐藏数据传输的延迟。
        #
        # 仅仅是把 Token ID 搬运到了 GPU 的连续显存池里，用于 Embedding 查找；真正的 KV Cache 页（Pages）分配
        # 会在调度器的其他地方进行。
        _slice = slice(cached_len, cached_len + chunk_size)
        # 1. 在 GPU 端的 Token Pool 中，切出第 table_idx 行的待写入区间
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        # 2. 从 CPU 端的 input_ids 中，切出本次需要传输的 token，并异步复制到 GPU
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )
```

当用户请求到达时，scheduler 会根据是否分块将其封装为一个 `Req` 对象或者`ChunkedReq`。详细可以分成三种情况：
1. 可以完全加入：预算和显存都很充足，请求被完整包装为普通 Req，加入 reqs。
2. 预算不足以处理，但支持“分块”（Chunked Prefill）：假设 GPU 这次最多还能算 1000 个 Token，但用户的 Prompt 有 3000 个 Token。系统不会拒绝或干等，而是把前 1000 个 Token 截断出来，包装成一个 ChunkedReq (分块请求) 加入本轮计算。同时，将这个没处理完的原请求放入 chunked_list 列表中，留待下次继续算剩下的 2000 个 Token。
3. 彻底加不进去了：连一小块都塞不下（可能显存满了或 Token 预算彻底耗尽），try_add_one 返回 None，触发 break 跳出循环，结束本轮的挑选。

`Req` 中最核心的字段是 `input_ids`——一个 CPU 上的 tensor，存储 tokenize 后的 token ID 序列：

```python
# python/minisgl/core.py
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

#### 6.1.1.3 更新队列与返回 Batch

```
if len(reqs) == 0:
    return None
self.pending_list = chunked_list + self.pending_list[len(reqs) :]
return Batch(reqs=reqs, phase="prefill")
```
- 兜底检查：如果一个请求都没能加进去（比如物理显存确实满了），返回 None。
- 更新等待队列（精髓所在）：
    - self.pending_list[len(reqs):]：已经完全处理完毕的请求会被移出队列。
    - chunked_list + ...：刚才被“分块截断”切了一半的请求，会被重新插回到等待队列的最头部。这样在 GPU 算完下一轮后，调度器会优先把这篇没读完的长文章继续读完。
- 打包返回：最终将挑出的这些 reqs 封装成一个 Batch 对象，标记阶段为 "prefill"，并交由调度器发往 GPU。

---

## 6.2 extend_len：理解增量计算

Mini-SGLang 引入了 `extend_len` 的概念，这是理解 prefill 过程的关键：

```python
# python/minisgl/core.py
@property
def extend_len(self) -> int:
    return self.device_len - self.cached_len
```

对于一个全新的请求，`cached_len = 0`，`extend_len` 等于整个输入长度。但当 RadixPrefixCache 命中了部分前缀时，`cached_len` 会大于 0，`extend_len` 就只包含未命中的那部分 token。这是一个精妙的简化：无论是否有 prefix cache 命中，后续流程都统一处理 `extend_len` 个 token，无需分支逻辑。

---

## 6.3 Prefill Batch 的准备

`PrefillManager` 负责从等待队列中挑选请求，组装成一个 prefill batch：

```python
# python/minisgl/scheduler/prefill.py
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
# python/minisgl/scheduler/scheduler.py
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

接下来，是如何分配所需的显存。这部分主要由 cache manager负责。
```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
        --> or decode_manager/schedule_next_batch
        --> _prepare_batch
            --> allocate_paged
                --> _allocate

    def allocate_paged(self, reqs: List[Req]) -> None:
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size) # 向上取整，已经在显存中的 Token 数；
            last_page = div_ceil(req.device_len, self.page_size) # 预期达到的 Token 数；
            if last_page > first_page:
                needed_pages += last_page - first_page # 计算需要新增的显示大小
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]
        return allocated
```
allocate_paged 函数计算 reqs 中每一个请求是否需要申请新的显存页，如果需要，则调用_allocate 函数。在_allocate 函数中，如果空闲显存页不足，调用self.prefix_cache.evict 函数驱逐长期未被使用的 kv cache。
最后，调用_write_page_table 函数，将分配的显存页按 token 展开，写入到 page_table（显存中）。

---

## 6.4 Positions Tensor 的构建

Prefill 需要为每个 token 提供正确的位置编码。`_make_positions` 函数为 batch 中每个请求生成从 `cached_len` 到 `device_len` 的位置索引：

```python
# python/minisgl/scheduler/scheduler.py
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

## 6.5 Attention 计算：FlashAttention-3

Prefill 阶段的 attention 使用 FlashAttention（版本 3 或 4，取决于 GPU 架构）。其 `forward` 方法先将 K、V 写入 cache，再调用 FlashAttention kernel：

```python
# python/minisgl/attention/fa.py
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
# python/minisgl/attention/fa.py
seqlens_q = [req.extend_len for req in reqs]   # query 长度 = extend_len
seqlens_k = [req.device_len for req in reqs]    # key 长度 = 完整序列长度
```

这里 `seqlens_q` 使用 `extend_len`（只计算新 token 的 query），而 `seqlens_k` 使用 `device_len`（attention 需要看到包括 cache 命中在内的完整 key 序列）。这就是为什么 prefix cache 能减少计算量：query 变短了，但 key 还是完整的。

---

## 6.6 KV Cache 写入与首 Token 采样

`forward_batch` 在模型前向计算完成后，调用 `complete_one` 更新每个请求的状态，然后进行 sampling：

```python
# python/minisgl/engine/engine.py
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
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE0MDM4OTg0MV19
-->