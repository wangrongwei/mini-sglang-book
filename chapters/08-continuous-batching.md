# 第 8 章 Continuous Batching 最小实现

> "Don't wait for the slowest; keep the fast ones moving." —— Continuous batching 的精髓在于不让已完成的请求占着 GPU 位置，也不让新请求苦等当前 batch 结束。

传统的 static batching 将一组请求打包在一起，必须等所有请求都生成完毕才能处理下一组。这意味着短序列必须等待长序列，GPU 在等待期间大量空转。Continuous batching 打破了这一限制：请求可以随时加入、随时退出，scheduler 在每一步动态决定哪些请求参与计算。Mini-SGLang 用极简的代码实现了这一核心机制。

---

## 8.1 Static Batching 的问题

假设一个 batch 包含 4 个请求，分别需要生成 10、50、200、5 个 token。在 static batching 下：

- 所有请求必须运行 200 步（最长序列的长度）
- 请求 4 在第 5 步就完成了，但它的 GPU 资源和 batch 位置要等到第 200 步才能释放
- 新到达的请求必须等当前 batch 全部结束后才能开始

这导致两个问题：**GPU 利用率低**（短序列完成后空转）和**延迟高**（新请求排队等待）。

## 8.2 Continuous Batching 的核心思想

Continuous batching 的解决方案很直观：

1. 每一步 decode 结束后，检查哪些请求已完成，立即移除
2. 移除后腾出的资源立即分配给等待中的新请求
3. 新请求经过 prefill 后加入 decode batch

关键在于 scheduler 需要在每一步都做出调度决策，而非以 batch 为单位做决策。

---

## 8.3 Mini-SGLang 的实现

### 调度主循环

Mini-SGLang 的 continuous batching 通过 scheduler 的主循环实现。每一轮循环执行四个步骤：

```
接收新请求 → _schedule_next_batch() → engine.forward_batch() → _process_last_data()
```

`_schedule_next_batch` 是 continuous batching 的核心调度点：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _schedule_next_batch(self) -> ForwardInput | None:
    batch = (
        self.prefill_manager.schedule_next_batch(self.prefill_budget)
        or self.decode_manager.schedule_next_batch()
    )
    return self._prepare_batch(batch) if batch else None
```

这段代码体现了一个关键的调度策略：**prefill 优先**。如果有等待 prefill 的请求，先安排 prefill；否则安排一轮 decode。这样新请求能尽快完成 prefill 并加入 decode 队列。

### 请求的加入与退出

当 `_process_last_data` 处理 prefill batch 的结果时，非 chunked 的请求会被 cache 并加入 decode 队列。`DecodeManager.filter_reqs` 将新完成 prefill 的请求合并到 `running_reqs` 中：

```python
# 文件: python/minisgl/scheduler/decode.py
def filter_reqs(self, reqs: Iterable[Req]) -> None:
    self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}
```

这个方法同时做了两件事：合并新请求，并过滤掉已完成的请求。`can_decode` 检查 `remain_len > 0`——如果一个请求在 prefill 阶段就因为 `output_len = 0` 而完成，它不会进入 decode 队列。

当 decode 步产生 EOS 或达到最大长度时，`_process_last_data` 立即将该请求从 `running_reqs` 中移除，释放 KV cache 资源：

```python
# 文件: python/minisgl/scheduler/scheduler.py
if finished and req not in self.finished_reqs:
    self.decode_manager.remove_req(req)
    self._free_req_resources(req)
```

下一轮调度时，`decode_manager.schedule_next_batch()` 自然就不包含已完成的请求，空出的 KV cache slots 可以分配给新请求。这就是 continuous batching 的运作机制——没有复杂的数据结构，仅靠 set 的增删实现动态 batch 管理。

---

## 8.4 Prefill 与 Decode 的混合

在完整的 SGLang 系统中，prefill 和 decode 可以在同一个 batch 中混合执行。Mini-SGLang 做了一个重要的简化：**每个 batch 要么全是 prefill，要么全是 decode**，不混合。

```python
# 文件: python/minisgl/core.py
@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
```

`phase` 字段是一个二选一的标记。这种简化大幅降低了 attention backend 的实现复杂度——不需要处理同一 batch 内既有长序列（prefill）又有单 token（decode）的情况。FlashAttention 只处理 prefill batch，FlashInfer 只处理 decode batch，职责清晰。

简化带来的代价是：当有新请求需要 prefill 时，正在 decode 的请求必须暂停一步。但由于 prefill 通常很快完成（一次或几次 chunked prefill），这个暂停对整体吞吐量的影响很小。

---

## 8.5 CUDA Graph 的 Padding 策略

Continuous batching 意味着 decode batch 的 size 在每步都可能变化（有请求完成退出，有请求 prefill 后加入）。但 CUDA graph 要求固定的 tensor shape。Mini-SGLang 通过 padding 解决这一矛盾：

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

预捕获的 batch size 列表为 `[1, 2, 4, 8, 16, 24, ...]`，粒度为 8。当 batch size 从 20 变为 18（2 个请求完成）时，都会 padding 到 24。当 size 变为 25（prefill 后加入新请求）时，padding 到 32。这种离散化策略限制了需要捕获的 graph 数量，同时保证计算浪费不超过 7 个 dummy request。

---

## 8.6 Overlap Loop 与 Normal Loop

Mini-SGLang 提供两种执行模式，`overlap_loop` 和 `normal_loop`，它们实现相同的 continuous batching 逻辑，但在 CPU-GPU 并行度上有所不同。

### Normal Loop

`normal_loop` 是最直观的实现——串行执行所有步骤：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def normal_loop(self) -> None:
    blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        ongoing_data = (forward_input, self._forward(forward_input))
    self._process_last_data(ongoing_data)
```

接收消息 -> 调度 -> 前向计算 -> 处理结果，一步步来。GPU 在执行前向计算时 CPU 空闲，CPU 在处理结果时 GPU 空闲。

### Overlap Loop

`overlap_loop` 通过**流水线**隐藏 CPU 延迟：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
    blocking = not (
        last_data is not None
        or self.prefill_manager.runnable
        or self.decode_manager.runnable
    )
    for msg in self.receive_msg(blocking=blocking):
        self._process_one_msg(msg)

    forward_input = self._schedule_next_batch()
    ongoing_data = None
    if forward_input is not None:
        with self.engine_stream_ctx:
            self.engine.stream.wait_stream(self.stream)
            ongoing_data = (forward_input, self._forward(forward_input))

    self._process_last_data(last_data)  # 处理上一轮的结果
    return ongoing_data
```

核心区别在于：`_process_last_data` 处理的是**上一轮**的结果（`last_data`），而当前轮的 `forward` 在引擎的 CUDA stream 上**并行执行**。当 GPU 执行当前 batch 的前向计算时，CPU 同时在处理上一个 batch 的 token 输出、更新请求状态、释放资源。两者使用不同的 CUDA stream，互不阻塞。

这种 overlap 策略特别适合 decode 阶段——单步 decode 的 GPU 计算很快，如果 CPU 端的处理不能及时完成，GPU 就会饥饿。通过流水线化，GPU 几乎不需要等待 CPU。

---

## 本章小结

1. **Static batching** 要求所有请求同时完成，导致 GPU 利用率低和延迟高；**continuous batching** 允许请求随时加入和退出。
2. Mini-SGLang 通过 `running_reqs` 集合的动态增删实现 continuous batching，每步调度时重新组装 batch。
3. **Prefill 优先**策略确保新请求尽快完成预填充并加入 decode 队列。
4. 每个 batch **不混合 prefill 和 decode**，简化了 attention backend 的实现，代价是 decode 请求偶尔暂停一步。
5. **CUDA graph padding** 将动态 batch size 对齐到预捕获的离散 size，在灵活性和性能之间取得平衡。
6. **Overlap loop** 通过双 CUDA stream 流水线化 CPU 处理和 GPU 计算，进一步提升吞吐量。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMzI5NzkwMDc2XX0=
-->