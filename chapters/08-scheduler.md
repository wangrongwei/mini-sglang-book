# 第 8 章 简化调度器设计

> "A good scheduler is invisible — requests flow in, tokens flow out, and nobody notices the orchestration in between." —— 调度器是推理引擎的大脑，它决定每一步谁来计算、谁来等待、谁该退场。

前面几章分别剖析了 prefill、decode 和 continuous batching 的机制。本章将这些线索汇聚到 scheduler——Mini-SGLang 系统的调度中枢。我们将完整追踪一个请求从进入等待队列到生成完毕的全生命周期，理解 `PrefillManager`、`DecodeManager` 和 `CacheManager` 如何协同工作。

---

## 8.1 等待队列与运行队列

Mini-SGLang 的调度器维护两类队列：

- **Waiting queue**（等待队列）：`PrefillManager.pending_list`，存放尚未开始 prefill 的请求
- **Running queue**（运行队列）：`DecodeManager.running_reqs`，存放已完成 prefill、正在逐 token 生成的请求

请求的状态转移路径为：

```
用户请求 → pending_list → prefill batch → running_reqs → 生成完毕 → 释放资源
```

这是一个单向流水线。`PrefillManager` 负责入口调度，`DecodeManager` 负责出口管理，`CacheManager` 在全程管理 KV cache 资源。

---

## 8.2 PrefillManager 的调度逻辑

### pending_list 与请求入队

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

### PrefillAdder 的逐条尝试

`schedule_next_batch` 创建一个 `PrefillAdder`，然后逐条尝试将 `pending_list` 中的请求加入本轮 batch：

```python
# 文件: python/minisgl/scheduler/prefill.py（概念代码）
def schedule_next_batch(self, token_budget: int) -> Batch | None:
    adder = PrefillAdder(
        token_budget=token_budget,
        reserved_size=...,
        cache_manager=self.cache_manager,
        table_manager=self.table_manager,
    )
    for msg in self.pending_list:
        adder.try_add_one(msg, ...)
    ...
    return Batch(reqs=adder.reqs, phase="prefill") if adder.reqs else None
```

`PrefillAdder.try_add_one` 是资源分配的核心，它按以下步骤检查每个请求：

1. **Table 空间检查**：是否还有可用的 table index
2. **Prefix cache 匹配**：通过 `CacheManager` 查找已有的 KV cache 前缀，确定 `cached_len`
3. **Token 预算检查**：该请求的 `extend_len` 是否在 `token_budget` 允许范围内
4. **Cache 空间估算**：预计需要的 page 数量是否在可用范围内

如果 token 预算不足以容纳一个完整请求，`PrefillAdder` 会创建 `ChunkedReq`——只处理预算允许的那部分 token。`ChunkedReq` 的 `can_decode` 返回 `False`，确保它不会被错误地加入 decode 队列：

```python
# 文件: python/minisgl/scheduler/prefill.py
class ChunkedReq(Req):
    def append_host(self, next_token):
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False
```

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

## 8.3 DecodeManager 的运行管理

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

## 8.4 _schedule_next_batch：调度决策

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

## 8.5 _prepare_batch：Tensor 组装

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

---

## 8.6 _process_last_data：后处理

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

---

## 8.7 请求状态转移全景

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
