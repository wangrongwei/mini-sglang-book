# 第 3 章 Request 与 Sequence

> "推理引擎的核心任务，是追踪每个请求'已经算到哪里了'。"

在 Mini-SGLang 中，一个用户请求从进入系统到输出完整回复，经历了多次状态转换。`Req` 类是这条生命线上最重要的数据结构——它不仅承载 token 序列本身，还精确记录了请求在 prefill/decode 流水线中的进度。本章深入剖析 `Req` 和 `SamplingParams` 的设计，以及请求从创建到完成的完整生命周期。

---

## 3.1 SamplingParams：用户意图的编码

每个请求都携带一份采样参数，决定了模型输出 logits 如何转换为下一个 token：

```python
# 文件: python/minisgl/core.py

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

几个值得注意的设计选择：

**默认贪心解码**：`temperature=0.0` 意味着默认选择概率最高的 token。这与 OpenAI API 的默认值（`temperature=1.0`）不同，反映了 Mini-SGLang 作为基础设施的定位——确定性输出更便于调试。

**`top_k=-1` 表示不启用**：当 `top_k` 为 -1 时，采样器不执行 top-k 过滤。这是一个常见的约定。

**`is_greedy` 的判定逻辑**：当温度 <= 0 或 top_k == 1 时，并且 top_p == 1.0（即不做 nucleus sampling），即为贪心。注意 `top_k == 1` 也是贪心——只保留最高概率的一个 token。

**`max_tokens` 控制输出上限**：这个参数直接影响 `Req` 的 `max_device_len`，从而决定 KV Cache 的预分配量。

---

## 3.2 Req 类：请求的运行时表示

`Req` 是调度器和引擎之间的核心接口。理解它的字段含义是读懂整个系统的前提：

```python
# 文件: python/minisgl/core.py

@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor       # CPU tensor，完整的 token 序列
    table_idx: int                # page_table 中的行索引
    cached_len: int               # 已写入 KV Cache 的 token 数
    output_len: int               # 最大输出 token 数（来自 max_tokens）
    uid: int                      # 用户请求唯一标识
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle # 前缀缓存句柄

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
```

### 三个长度字段的语义

`Req` 内部维护了三个关键长度，它们的关系构成了请求进度的完整描述：

```
|<---- cached_len --->|<--- extend_len --->|<--- remain_len --->|
|     已在 KV Cache    |   本轮需要计算的    |    未来还要生成的    |
0                   cached_len          device_len          max_device_len
```

- **`cached_len`**：KV Cache 中已经存在的 token 数。对于 prefill 阶段有前缀缓存命中的情况，`cached_len > 0`；否则为 0
- **`device_len`**：当前已知的总 token 数（input + 已生成的 output）。在 `__post_init__` 中初始化为 `len(input_ids)`
- **`max_device_len`**：理论最大长度 = `len(input_ids) + output_len`。这决定了需要预留的 KV Cache slot 上限

由此派生出两个关键属性：

```python
# 文件: python/minisgl/core.py

@property
def remain_len(self) -> int:
    return self.max_device_len - self.device_len

@property
def extend_len(self) -> int:
    return self.device_len - self.cached_len
```

- **`extend_len`**：本轮 forward 需要实际计算 attention 的 token 数。在 prefill 阶段等于 `input_len - cached_len`，在 decode 阶段等于 1
- **`remain_len`**：距离生成上限还剩多少 token。当 `remain_len == 0` 时，请求因达到 `max_tokens` 而结束

---

## 3.3 table_idx：请求在 Page Table 中的位置

`table_idx` 是理解 Mini-SGLang 内存管理的关键字段。全局 `Context` 持有一个二维 `page_table` 张量：

```python
# 概念示意（实际在 engine.py 中分配）
page_table = torch.zeros(max_running_reqs, max_seq_len, dtype=torch.int32, device="cuda")
```

每个 `Req` 通过 `table_idx` 占据 `page_table` 的一行。这一行记录了该请求的每个 token position 对应的 KV Cache 物理 slot：

```
page_table[req.table_idx] = [slot_0, slot_1, slot_2, ..., slot_N, 0, 0, ...]
                              ↑                              ↑
                          position 0 的 KV          position N 的 KV
```

`table_idx` 的分配和回收由 `TableManager` 管理：

```python
# 文件: python/minisgl/scheduler/table.py

class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
```

`TableManager` 维护一个简单的 free list。`token_pool` 是与 `page_table` 形状相同的张量，存储每个位置的 token ID（GPU 侧），用于 CUDA Graph 回放时直接读取输入。

---

## 3.4 请求生命周期

一个请求从用户发出到回复完成，经历以下阶段：

```
    ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────┐
    │ Pending  │────→│ Prefill  │────→│ Decode Loop  │────→│ Finished │
    └──────────┘     └──────────┘     └──────────────┘     └──────────┘
      PendingReq        Req              Req                  (释放)
```

### 阶段 1：Pending

用户请求经过 Tokenizer 转换为 token IDs 后，以 `PendingReq` 的形式进入 `PrefillManager.pending_list`：

```python
# 文件: python/minisgl/scheduler/utils.py (概念)
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor         # CPU tensor
    sampling_params: SamplingParams
    chunked_req: Req | None = None  # 用于 chunked prefill
```

此时请求尚未分配 `table_idx`，也没有 KV Cache 资源。

### 阶段 2：Prefill

`PrefillAdder.try_add_one()` 为请求分配资源并创建 `Req` 对象：

```python
# 文件: python/minisgl/scheduler/prefill.py

def _add_one_req(self, pending_req, cache_handle, table_idx, cached_len) -> Req:
    remain_len = pending_req.input_len - cached_len
    chunk_size = min(self.token_budget, remain_len)
    is_chunked = chunk_size < remain_len
    CLS = ChunkedReq if is_chunked else Req
    self.token_budget -= chunk_size
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

注意 **Chunked Prefill** 机制：如果 prompt 太长超出 `token_budget`，只处理前 `chunk_size` 个 token，创建 `ChunkedReq`（一个禁止 decode 的特殊子类），剩余部分留到下一轮。

Prefill 完成后，`extend_len` 个 token 的 KV 被写入 Cache，`cached_len` 被更新为 `device_len`。

### 阶段 3：Decode Loop

Prefill 完成的 `Req`（非 `ChunkedReq`）进入 `DecodeManager.running_reqs`。每轮 decode：

```python
# 文件: python/minisgl/core.py

def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1

def append_host(self, next_token: torch.Tensor) -> None:
    self.input_ids = torch.cat([self.input_ids, next_token])
```

`complete_one()` 标记本轮生成的 token 已写入 KV Cache（`cached_len = device_len`），然后 `device_len += 1` 表示序列长度增加了一个 token。`append_host()` 将新 token 追加到 CPU 侧的 `input_ids`，保持完整序列记录。

每轮 decode 后检查终止条件：

```python
# 文件: python/minisgl/core.py

@property
def can_decode(self) -> bool:
    return self.remain_len > 0
```

当 `remain_len == 0`（达到 `max_tokens`）或生成了 EOS token（且 `ignore_eos=False`），请求进入 Finished 阶段。

### 阶段 4：Finished

请求完成后，调度器执行清理：
1. 调用 `CacheManager.cache_req(req, finished=True)` 将 KV Cache 数据插入 Radix Prefix Cache（供未来请求复用）
2. 释放 `table_idx` 回 `TableManager`
3. 将生成结果通过消息队列返回给 Tokenizer/Frontend

---

## 3.5 Decode 阶段的 Req 状态变化示例

以一个 `input_ids = [1, 2, 3]`、`max_tokens = 3` 的请求为例：

| 时刻 | cached_len | device_len | max_device_len | extend_len | remain_len | 阶段 |
|------|-----------|-----------|---------------|-----------|-----------|------|
| 创建 | 0 | 3 | 6 | 3 | 3 | Prefill |
| Prefill 后 | 3 | 3 | 6 | 0 | 3 | 等待 Decode |
| Decode 1 | 3→3 | 3→4 | 6 | 1 | 2 | Decode |
| Decode 1 完成 | 4 | 4 | 6 | 0 | 2 | — |
| Decode 2 | 4→4 | 4→5 | 6 | 1 | 1 | Decode |
| Decode 2 完成 | 5 | 5 | 6 | 0 | 1 | — |
| Decode 3 | 5→5 | 5→6 | 6 | 1 | 0 | Decode |
| Decode 3 完成 | 6 | 6 | 6 | 0 | 0 | Finished |

每次 `complete_one()` 调用，`cached_len` 追上 `device_len`，然后 `device_len` 向前推进 1。当 `device_len == max_device_len` 时，`remain_len` 归零，请求结束。

---

## 3.6 `eq=False` 的设计意图

注意 `Req` 使用了 `@dataclass(eq=False)`：

```python
@dataclass(eq=False)
class Req:
    ...
```

这意味着 `Req` 对象使用默认的 `id()` 比较（即引用相等），而不是字段值比较。这是因为：

1. `Req` 的字段在生命周期内不断变化（`cached_len`、`device_len` 等），基于值的相等性没有意义
2. 同一个请求在系统中只有一个 `Req` 实例，用引用相等即可
3. 避免比较 `torch.Tensor` 字段导致的意外行为

---

## 本章小结

1. `SamplingParams` 编码用户的采样偏好，默认使用贪心解码（`temperature=0.0`），`is_greedy` 属性综合 `temperature`、`top_k`、`top_p` 三个维度判定
2. `Req` 通过三个长度字段（`cached_len`、`device_len`、`max_device_len`）精确追踪请求进度，派生出 `extend_len`（本轮计算量）和 `remain_len`（剩余生成量）
3. `table_idx` 将请求映射到 `page_table` 的一行，建立 token position 到 KV Cache 物理 slot 的映射
4. 请求生命周期经历 Pending → Prefill → Decode Loop → Finished 四个阶段，`complete_one()` 是 decode 循环的核心状态推进方法
5. Chunked Prefill 通过 `ChunkedReq` 子类实现，将长 prompt 分块处理，控制单轮的计算量和内存压力
