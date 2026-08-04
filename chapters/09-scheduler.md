# 第 9 章 简化调度器设计

> "A good scheduler is invisible — requests flow in, tokens flow out, and nobody notices the orchestration in between." —— 调度器是推理引擎的大脑，它决定每一步谁来计算、谁来等待、谁该退场。

前面几章分别剖析了 prefill、decode 和 continuous batching 的机制。本章将这些线索汇聚到 scheduler——Mini-SGLang 系统的调度中枢。我们将完整追踪一个请求从进入等待队列到生成完毕的全生命周期，理解 `PrefillManager`、`DecodeManager` 和 `CacheManager` 如何协同工作。

---

## 9.1 初始化
后端调度器初始化包括：
```python
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.token_pool = self.table_manager.token_pool
        self.prefill_budget = config.max_extend_tokens
```
可以将初始化过程分为四个部分来理解：
### 9.1.1 引擎初始化 (Engine Initialization)
```
self.engine = Engine(config)
```
这里初始化了底层的推理引擎 (Engine)。Engine 负责实际的模型计算（前向传播）。
### 9.1.2 CUDA 并发流设置 (CUDA Stream Setup)
```
# use another stream to overlap metadata processing with computation
self.device = self.engine.device
self.stream = torch.cuda.Stream(device=self.device)
self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
torch.cuda.set_stream(self.stream)
```
这是非常关键的性能优化代码。
- 这里为 Scheduler 专门创建了一个新的 CUDA 流 (self.stream)，并将当前默认流切换为它。
- 重叠元数据处理与计算：当 Engine 在其自己的流 (self.engine.stream) 中疯狂进行矩阵乘法（计算模型前向传播）时，Scheduler 可以在这个新的流中同时准备下一批请求的元数据（比如分配内存、构建索引、拷贝输入数据等）。这种 Overlap（重叠）能显著降低延迟并提高 GPU 利用率。
### 9.1.3 子管理器初始化 (Manager Initialization)
```
# initialize other managers
self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
self.cache_manager = CacheManager(
    self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
)
self.decode_manager = DecodeManager(config.page_size)
self.prefill_manager = PrefillManager(
    self.cache_manager, self.table_manager, self.decode_manager
)
```
调度器将复杂的调度任务拆分给了几个专业的包工头：
- TableManager (页表管理器)：一个二维的 PyTorch Tensor，形状是 (最大并发请求数 + 1, 最大序列长度)，用来存储每个请求对应的 KV Cache 物理位置映射（页表）。最大序列长度受config.max_seq_len 以及实际显存大小约束。
- CacheManager (缓存管理器)：管理显存中实际的 KV Cache 物理块（Pages）。它会根据 cache_type（如是否使用 Radix Tree）来进行内存块的分配和回收。
- DecodeManager (解码管理器)：专门负责“解码阶段”（每次吐出一个 Token）的调度逻辑。
- PrefillManager (预填充管理器)：专门负责“预填充阶段”（处理用户长 Prompt）的调度。因为预填充需要分配大量初始 Cache，并且完成后要转交给解码阶段，所以它在初始化时传入了另外三个管理器作为依赖。
### 9.1.4 快捷别名与状态变量 (Aliases and States)
```
# some alias for easy access
self.finished_reqs: Set[Req] = set()
self.tokenizer = load_tokenizer(config.model_path)
self.eos_token_id = self.tokenizer.eos_token_id
self.token_pool = self.table_manager.token_pool
self.prefill_budget = config.max_extend_tokens
```
最后初始化了一些常用的状态和快捷方式：
- finished_reqs：用来记录当前已经生成完毕（结束）的请求。
- tokenizer / eos_token_id：加载分词器，并提取出“结束符 (EOS)”，用于判断请求是否应该停止生成。
- token_pool：通过别名直接指向 TableManager 中的 Token 内存池，方便后续快速访问。
- prefill_budget：设置每次 Prefill 操作允许处理的最大 Token 数量上限，这是为了防止单个长文本请求把 GPU 显存或计算资源一次性耗尽（Chunked Prefill 的基础）。


## 9.2 等待队列与运行队列

Mini-SGLang 的调度器维护两类队列：

- **Waiting queue**（等待队列）：`PrefillManager.pending_list`，存放尚未开始 prefill 的请求
- **Running queue**（运行队列）：`DecodeManager.running_reqs`，存放已完成 prefill、正在逐 token 生成的请求

请求的状态转移路径为：

```
用户请求 → pending_list → prefill batch → running_reqs → 生成完毕 → 释放资源
```

这是一个单向流水线。`PrefillManager` 负责入口调度，`DecodeManager` 负责出口管理，`CacheManager` 在全程管理 KV cache 资源。

---

## 9.3 PrefillManager 的调度逻辑

### 9.3.1 pending_list 与请求入队

新请求到达时，`PrefillManager.add_one_req` 将其加入 `pending_list`：

```python
# 文件: python/minisgl/scheduler/prefill.py（概念代码）
class PrefillManager:
    pending_list: List[UserMsg]
    cache_manager: CacheManager
    decode_manager: DecodeManager

    def add_one_req(self, msg: UserMsg) -> None:
        self.pending_list.append(msg)

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0
```

`runnable` 属性告诉 scheduler 是否有请求在等待 prefill。这个信号直接影响 `_schedule_next_batch` 的决策。

### 9.3.2 PrefillAdder 的逐条尝试

`schedule_next_batch` 创建一个 `PrefillAdder`，然后逐条尝试将 `pending_list` 中的请求加入本轮 batch：

```python
# python/minisgl/scheduler/prefill.py
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

`PrefillAdder.try_add_one` 是资源分配的核心，它按以下步骤检查每个请求：

1. **Table 空间检查**：是否还有可用的 table index
2. **Prefix cache 匹配**：通过 `CacheManager` 查找已有的 KV cache 前缀，确定 `cached_len`
3. **Token 预算检查**：该请求的 `extend_len` 是否在 `token_budget` 允许范围内
4. **Cache 空间估算**：预计需要的 page 数量是否在可用范围内

#### 9.3.2.1 预算与资源预估
```
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
```
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

#### 9.3.2.2 分块预填充（chunked prefill）
在调度层的最开始，会判断一个请求是否属于 chunked_req类型，如果是，则会加入到chunked_list中。
```
reqs: List[Req] = []
chunked_list: List[PendingReq] = []
for pending_req in self.pending_list:
    if req := adder.try_add_one(pending_req):
        pending_req.chunked_req = None
        if isinstance(req, ChunkedReq):
            pending_req.chunked_req = req
            chunked_list.append(pending_req)
        reqs.append(req)
    ...
```
因为有分块预填充（Chunked Prefill）机制，如果上一次调度时用户的长文本没算完（比如被截断了一半），它的状态会被保存在 pending_req.chunked_req 中。这里优先判断它是不是一个“历史遗留”的半截请求：如果是，不需要重新去申请内存或匹配 Radix Tree 前缀，直接拿出它之前算了一半的 cache_handle（缓存句柄）、table_idx（物理页表位置）和 cached_len（已经算到了第几个字），传给 _add_one_req 继续往下算。
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
继续往下，对于一个刚到的新请求，会直接进入_try_allocate_one函数进行资源预估和 prefix cache 匹配。_try_allocate_one函数在上一小节已经介绍，这里直接进入_add_one_req函数。

如果 token 预算不足以容纳一个完整请求，`PrefillAdder` 会创建 `ChunkedReq`——只处理预算允许的那部分 token。如果上一次调度时用户的长文本没算完（比如被截断了一半），它的状态会被保存在 pending_req.chunked_req 中。这里优先判断它是不是一个“历史遗留”的半截请求：如果是，不需要重新去申请内存或匹配 Radix Tree 前缀，直接拿出它之前算了一半的 cache_handle（缓存句柄）、table_idx（物理页表位置）和 cached_len（已经算到了第几个字），传给 _add_one_req 继续往下算。

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
这里会发生三种情况：
1. 可以完全加入：预算和显存都很充足，请求被完整包装为普通 Req，加入 reqs。
2. 预算不足以处理，但支持“分块”（Chunked Prefill）：假设 GPU 这次最多还能算 1000 个 Token，但用户的 Prompt 有 3000 个 Token。系统不会拒绝或干等，而是把前 1000 个 Token 截断出来，包装成一个 ChunkedReq (分块请求) 加入本轮计算。同时，将这个没处理完的原请求放入 chunked_list 列表中，留待下次继续算剩下的 2000 个 Token。
3. 彻底加不进去了：连一小块都塞不下（可能显存满了或 Token 预算彻底耗尽），try_add_one 返回 None，触发 break 跳出循环，结束本轮的挑选。

`ChunkedReq` 的 `can_decode` 返回 `False`，确保它不会被错误地加入 decode 队列：

```python
# 文件: python/minisgl/scheduler/prefill.py
class ChunkedReq(Req):
    def append_host(self, next_token):
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False
```

#### 9.3.2.3 更新队列与返回 Batch
在前面schedule_next_batch函数末尾，会返回包装好的 Batch：
```python
if len(reqs) == 0:
    return None
self.pending_list = chunked_list + self.pending_list[len(reqs) :]
return Batch(reqs=reqs, phase="prefill")
```
这里主要有：
- 兜底检查：如果一个请求都没能加进去（比如物理显存确实满了），返回 None。
- 更新等待队列（精髓所在）：
    - self.pending_list[len(reqs):]：已经完全处理完毕的请求会被移出队列。
    - chunked_list + ...：刚才被“分块截断”切了一半的请求，会被重新插回到等待队列的最头部。这样在 GPU 算完下一轮后，调度器会优先把这篇没读完的长文章继续读完。
- 打包返回：最终将挑出的这些 reqs 封装成一个 Batch 对象，标记阶段为 "prefill"，并交由调度器发往 GPU。


### reserved_size 的考量

`PrefillAdder` 的 `reserved_size` 参数预留了 decode 阶段所需的资源。在估算 cache 需求时，需要考虑当前 decode 中的请求未来还会消耗多少 page。`DecodeManager.inflight_tokens` 提供了这个估算：

```python
# 文件: python/minisgl/scheduler/decode.py
@property
def inflight_tokens(self) -> int:
    tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
    return sum(req.remain_len for req in self.running_reqs) + tokens_reserved
```

这保证了新加入 prefill 的请求不会抢占正在 decode 的请求所需的资源。

---

## 9.4 DecodeManager 的运行管理

`DecodeManager` 的设计非常简洁——一个 set 存储所有正在 decode 的请求：

```python
# 文件: python/minisgl/scheduler/decode.py
@dataclass
class DecodeManager:
    page_size: int
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            return None
        return Batch(reqs=list(self.running_reqs), phase="decode")
```

`filter_reqs` 在 prefill 完成后被调用，将新请求合并到 `running_reqs` 中。`remove_req` 在请求完成或被 abort 时调用。`schedule_next_batch` 直接将整个 `running_reqs` 打包成 batch——没有优先级排序、没有选择性调度，所有正在 decode 的请求一起参与。

这是 Mini-SGLang 相对于完整 SGLang 的一个重要简化。完整系统需要考虑内存压力下的请求 preemption（抢占）、优先级调度等。Mini-SGLang 假设所有 decode 请求都能同时运行，大幅简化了实现。

---

## 9.5 _schedule_next_batch：调度决策

调度器每轮的核心决策只有一行：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

利用 Python 的 `or` 短路求值：先尝试 prefill，如果 `PrefillManager` 返回非空 batch 就执行 prefill；否则尝试 decode。如果两个 manager 都没有可运行的请求，返回 `None`。

这种 "prefill first" 策略意味着：

- 新请求能以最低延迟开始生成（Time to First Token 优化）
- 正在 decode 的请求会因 prefill 而暂停一步
- 在高负载下，prefill 和 decode 交替执行

注释中提到 `TODO: support other policies: e.g. DECODE first`，说明这只是最基本的策略，完整系统支持更灵活的调度。

---

## 9.6 _prepare_batch：Tensor 组装

调度决策产出 `Batch` 对象后，`_prepare_batch` 将其转化为 GPU 可执行的 tensor 集合：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    self.engine.graph_runner.pad_batch(batch)        # 1. CUDA graph padding
    self.cache_manager.allocate_paged(batch.reqs)    # 2. 分配 KV cache pages
    batch.positions = _make_positions(batch, self.device)  # 3. 位置编码
    input_mapping = _make_input_tuple(batch, self.device)  # 4. 输入映射
    write_mapping = _make_write_tuple(batch, self.device)  # 5. 写入映射
    batch.out_loc = self.engine.page_table[input_mapping]  # 6. 查找物理位置
    self.engine.attn_backend.prepare_metadata(batch)       # 7. Attention 元数据
    return ForwardInput(
        batch=batch,
        sample_args=self.engine.sampler.prepare(batch),    # 8. Sampling 参数
        input_tuple=input_mapping,
        write_tuple=write_mapping,
    )
```

这八步按顺序完成了从逻辑请求到物理 tensor 的转换：

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | pad_batch | 对齐到 CUDA graph batch size |
| 2 | allocate_paged | 为新 token 分配 KV cache page |
| 3 | _make_positions | 生成位置编码 tensor |
| 4 | _make_input_tuple | 建立 table_idx -> position 的映射 |
| 5 | _make_write_tuple | 建立 output token 的写入位置 |
| 6 | page_table 查找 | 将逻辑位置转换为 KV cache 物理位置 |
| 7 | prepare_metadata | 构建 attention kernel 所需的元数据 |
| 8 | sampler.prepare | 准备 sampling 参数（temperature 等） |

这里涉及一部分分配 kv cache page 的流程，必须介绍一下。
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
allocate_paged 函数计算 reqs 中每一个请求是否需要申请新的显存页，如果需要，则调用_allocate 函数。在_allocate 函数中，如果空闲显存页不足，调用self.prefix_cache.evict 函数驱逐长期未被使用的 kv cache（在第 4 章kvcache介绍中有这部分内容）。
最后，调用_write_page_table 函数，将分配的显存页按 token 展开，写入到 page_table（显存中）。

---

## 9.7 _process_last_data：后处理

前向计算完成后，`_process_last_data` 负责收割结果并驱动状态转移：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _process_last_data(self, last_data):
    if last_data is None:
        return
    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()  # 等待 GPU -> CPU 拷贝完成

    reply: List[DetokenizeMsg] = []
    new_finished_reqs: Set[Req] = set()
    with self.cache_manager.lazy_free_region():
        for i, req in enumerate(batch.reqs):
            if isinstance(req, ChunkedReq):
                continue  # chunked 请求跳过 sampling 结果
            next_token = next_tokens_cpu[i]
            req.append_host(next_token.unsqueeze(0))
            finished = not req.can_decode
            if not req.sampling_params.ignore_eos:
                finished |= next_token == self.eos_token_id
            reply.append(DetokenizeMsg(uid=req.uid, next_token=..., finished=finished))

            if finished and req not in self.finished_reqs:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
                new_finished_reqs.add(req)
            elif batch.is_prefill:
                self.cache_manager.cache_req(req, finished=False)

    self.finished_reqs = new_finished_reqs
    self.send_result(reply)
```

几个关键细节：

- **`copy_done.synchronize()`**：前向计算将 next_tokens 从 GPU 异步拷贝到 CPU，这里同步等待拷贝完成
- **`lazy_free_region`**：使用 context manager 延迟释放 cache，避免释放后立即被新分配覆盖（与 overlap loop 配合使用）
- **ChunkedReq 跳过**：分块请求还未完成 prefill，不应处理其 sampling 结果
- **Prefill 后 cache**：非 chunked 的 prefill 请求调用 `cache_req` 将前缀写入 prefix cache，供后续请求复用
- **`finished_reqs` 防重复释放**：overlap 模式下同一请求可能被处理两次，用 set 防护

这个阶段，有一个重要的步骤，就是将已经结束的请求的 kv 保存在 prefix cache （即 kv cache 树） 中。
```
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
        --> decode_manager/schedule_next_batch
        --> _prepare_batch
            --> allocate_paged
                --> _allocate
    --> _process_last_data
        --> _free_req_resources
          --> cache_req
            --> insert_prefix
        --> cache_req
            --> insert_prefix

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        old_handle = req.cache_handle
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle
            self.lock(new_handle)
```
对于 radix 类型的 prefix cache，缓存一个请求的 kv 流程：
```python
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            new_node = RadixTreeNode(self.key_fn)
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            new_node.set_parent(node)
            self.evictable_size += new_node.length
            node = new_node
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))
```

---

## 9.8 重叠调度（overlap schedule）

```python
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
        --> decode_manager/schedule_next_batch
    --> _forward
    --> _process_last_data
```
举例说明重叠调度。假如用户 A 问：“***请给我写一篇一万字的小说。***” 系统为了防止滥用，或用户自己设置了最多只能输出 5 个字 (max_new_tokens = 5)。 当前状态：模型已经输出了“从”、“前”、“有”、“座”。（已经 4 个字了）。 本轮 GPU 计算结果：GPU 正聊得起劲，算出来的下一个字是“山”。所以传给 CPU 的 next_tokens_cpu 里，用户 A 对应的 token 是 105（代表“山”）。

在_process_last_data函数中，会在下一轮处理用户 A 的请求返回的 token：
```python
# 1. 拿到本轮属于用户 A 的字：张量 Tensor(105)
next_token = next_tokens_cpu[i] 

# 2. 把 105 ("山") 存进用户 A 的历史记录本里。现在记录本里有 5 个字了。
req.append_host(next_token.unsqueeze(0))

# 3. 把张量变成普通数字 105。
next_token = int(next_token.item()) 

# 4. 判断用户 A 还能接着聊吗？
# 注意！系统发现用户 A 已经输出了 5 个字，达到了上限！req.can_decode 判定为 False (不准聊了)。
# finished = not False，所以此时 finished 变成了 True！
finished = not req.can_decode 

# 5. 用户 A 也没有设置“强制忽略结束符”。
if not req.sampling_params.ignore_eos:
    # 6. 检查当前的字是不是结束符：当前字是 105("山")，不是 999。
    # 但没关系，因为前面的 finished 已经是 True 了，True | False 依然是 True。
    finished |= next_token == self.eos_token_id 

# 7. 呼叫快递员 (DetokenizeMsg)：
# "嘿！给用户 A (uid) 送去最新产出的字 105（"山"），顺便告诉他，输出被强行掐断了 (finished=True)！"
# 前端收到消息，打字机停留在“从前有座山”，虽然句子没说完，但光标也停止了闪烁。
reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))
```

下一行代码很关键，为什么需要？

```python
self.finished_reqs = new_finished_reqs
```

这里记录 self.finished_reqs = new_finished_reqs 的核心目的，是为了防止显存资源被“重复释放（Double Free）”。推理引擎为了不让 GPU 闲着，使用 重叠调度（Overlap Scheduling）的方式：当 GPU 正在拼命计算 第 N 批次（Batch N）的数据时，CPU 并没有在旁边干等，而是提前开始准备 第 N+1 批次（Batch N+1）的请求名单：
- 假设“请求A”在 Batch N 的计算中恰好结束了。
- 但是，因为 CPU 提前排班，它在排 Batch N+1 的时候，还没收到“请求A已结束”的通知，所以把“请求A”又放进了 Batch N+1 里。
- 当 CPU 回过头来处理 Batch N 的结果时，它发现请求A结束了，于是释放了请求A占用的显存（self._free_req_resources(req)）。
- 紧接着，当 CPU 处理 Batch N+1 的结果时，它再次发现请求A结束了，如果没有任何防范，它会去第二次释放请求A的显存！ 这会导致灾难性的 Bug（比如把刚刚分给别人的新显存给释放了，即 Use-After-Free）。
假设用户 A 的提问是：“天王盖地虎的下一句是什么？” 大模型要回答：“宝塔镇河妖。<结束>” (共 6 个 Token)。这个请求在引擎里的实际流程是这样的：
- 第 1 轮
    - CPU：创建 Batch 1（把请求 A 放进去），发给 GPU。不等 GPU 算完，立刻准备 第 2 轮。并把请求 A 放进 第 2 轮队列，发给 GPU。
    - GPU：开始算 Batch 1，试图预测第 1 个字。
- 第 2 轮
    - GPU：算完 Batch 1，把 “宝” 字传回 CPU。开始算排好队的 Batch 2。
    - CPU：结算 Batch 1，拿到“宝”字，请求 A 没结束。准备第 3 轮：把请求 A 放入 第 3 轮 队列，发给 GPU。
- 第 3、4、5、6 轮 ...
- 第 7 轮
    - GPU：算完 Batch 6，把句号 “。” 传回 CPU。开始算 Batch 7（**第 7 轮算的就是附加“。”的请求 A**）。
    - CPU：结算 Batch 6，拿到句号“。”。因为“。”还不是结束符，请求 A 依然视为未结束。准备第 8 轮，CPU 再次把请求 A 放入了第 8 轮名单，并交给了 GPU 排队（**其实这里已经不需要了，但是 CPU 未能感知 GPU 已经算出的结束符**）。
- 第 8 轮
    - GPU：算完 Batch 7，把 <结束符> 传回 CPU。立刻处理包含请求 A 的第 8 轮名单，开始算请求 A 下一个 Token。
    - CPU：结算 Batch 7，收到 <结束符> （**现在 CPU 才感知到请求 A 结束**）！CPU 宣布请求 A 彻底结束。立刻释放请求 A 的显存资源，并把请求 A 记入 self.finished_reqs (**但 GPU 还在并行处理第 8 轮**)。然后，准备第 9 轮，后续名单已经没有请求 A。
- 第 9 轮
    - GPU：算完 Batch 8，把一堆因为盲目计算而产生的废弃结果传回给 CPU。开始算没有请求 A 的 Batch 9。
    - CPU：结算 Batch 8，CPU 遍历 Batch 8 的名单，发现请求 A！（**因为在第 7 轮排班时，没预料到结束符，正常塞给了 GPU**）。
        - ⅰ. 代码走到 if finished and req not in self.finished_reqs: 时，发现请求 A 在上一轮结算时已经被放进 self.finished_reqs！说明显存已经释放，可直接跳过！
只要大模型还没吐出 <结束符>，哪怕这个请求已经处理了 100 次，在第 101 次排班时，CPU 会加上新吐出的信息附加，然后把它加进当次计算的名单，发给 GPU 继续算下一个字。

> 在连续的解码阶段，CPU 发给 GPU 的请求 A，不再是包含 Token 的张量，而是一个类似于指针和指令的包裹。
> - 第 1 轮 CPU 发给 GPU 的指令（请求 A）： 现在去查 12 号显存块（请求 A 的 KV Cache），把里面的状态取出来，给我算下一个 Token。
> - 第 2 轮 CPU 发给 GPU 的指令（长得确实一模一样）： “GPU，继续去查 12 号显存块。至于输入数据，直接用上一步刚刚算出来的那个 Token，给我算再下一个 Token。”

---

## 9.8 请求状态转移全景

将所有组件串联，一个请求的完整生命周期如下：

```
1. 用户发送请求
   ↓
2. PrefillManager.add_one_req() → 加入 pending_list
   ↓
3. _schedule_next_batch() → PrefillManager.schedule_next_batch()
   PrefillAdder.try_add_one() 检查资源并分配
   ↓
4. _prepare_batch() → 分配 KV cache、构建 tensor
   ↓
5. engine.forward_batch() → 模型前向 + sampling
   Req.complete_one() → cached_len = device_len, device_len += 1
   ↓
6. _process_last_data() → append_host(next_token)
   如果是 prefill batch: cache_req() 写入 prefix cache
   DecodeManager.filter_reqs() 将请求加入 running_reqs
   ↓
7. _schedule_next_batch() → DecodeManager.schedule_next_batch()
   所有 running_reqs 打包成 decode batch
   ↓
8. 重复步骤 4-7，每步生成一个 token
   ↓
9. can_decode == False 或遇到 EOS
   DecodeManager.remove_req() → _free_req_resources()
   ↓
10. 结果发送给用户
```

整个流程中，调度器扮演的角色是**资源分配者**和**状态管理者**。它不关心模型如何计算 attention、如何 sample token——这些是 engine 的职责。调度器只需要知道：谁在等待、谁在运行、谁该退出、资源够不够。

---

## 本章小结

1. **两级队列架构**：`pending_list`（等待 prefill）和 `running_reqs`（正在 decode）构成请求流转的两个阶段。
2. **PrefillAdder** 通过 token budget、cache 空间、table 可用性三重检查，逐条决定请求是否加入本轮 prefill batch，支持 chunked prefill 处理超长输入。
3. **DecodeManager** 极度简化——将所有运行中请求直接打包，不做优先级调度或 preemption，这是 Mini-SGLang 相对于完整系统最显著的简化之一。
4. **`_schedule_next_batch`** 用一行 `or` 表达式实现 "prefill first" 策略。
5. **`_prepare_batch`** 按八步流程将逻辑 batch 转化为 GPU tensor，是调度器与 engine 之间的桥梁。
6. **`_process_last_data`** 处理生成结果、驱动状态转移、管理资源释放，是 continuous batching 的闭环关键。
7. 请求的完整生命周期是一条**单向流水线**：waiting -> prefill -> decode -> finish，调度器在每一步分配资源并管理状态。
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDU2NDA0ODM2XX0=
-->