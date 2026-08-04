# 第 15 章 扩展 Mini-SGLang

> "The best way to learn is to do." —— Richard Feynman

Mini-SGLang 的代码结构清晰、模块化程度高，非常适合作为学习和实验的平台。本章通过三个动手实例——添加新的 Sampling 策略、为调度器增加请求优先级、以及添加监控指标端点——展示如何在 Mini-SGLang 上进行功能扩展，帮助读者将源码阅读的理解转化为工程实践能力。

---

## 15.1 实例一：添加 Min-P Sampling

### 背景

Mini-SGLang 目前支持 Greedy、Top-K 和 Top-P 三种采样策略（见 `python/minisgl/engine/sample.py`）。Min-P 是一种较新的采样方法：给定概率最高的 Token 概率 $p_{\max}$，只保留概率不低于 $p_{\max} \times \text{min\_p}$ 的 Token。它比 Top-P 更稳定，不需要排序操作。

### 第一步：扩展 SamplingParams

在 `python/minisgl/core.py` 中的 `SamplingParams` 数据类添加 `min_p` 字段：

```python
# 文件: python/minisgl/core.py
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0          # 新增字段
    max_new_tokens: int = 1024
```

### 第二步：扩展 Sampler.prepare()

在 `python/minisgl/engine/sample.py` 的 `Sampler.prepare()` 方法中收集 `min_p` 参数：

```python
# 文件: python/minisgl/engine/sample.py
@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None
    top_p: torch.Tensor | None
    min_p: torch.Tensor | None   # 新增字段

class Sampler:
    def prepare(self, batch: Batch) -> None:
        # ... 现有的 temperature / top_k / top_p 收集逻辑 ...
        min_p_list = [r.sampling_params.min_p for r in batch.reqs]
        has_min_p = any(v > 0.0 for v in min_p_list)
        self.args = BatchSamplingArgs(
            temperatures=...,
            top_k=...,
            top_p=...,
            min_p=make_device_tensor(min_p_list) if has_min_p else None,
        )
```

### 第三步：实现 Min-P 过滤逻辑

在 `sample_impl()` 函数中添加 Min-P 过滤：

```python
# 文件: python/minisgl/engine/sample.py
def sample_impl(logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
    # 温度缩放
    probs = torch.softmax(logits / args.temperatures.unsqueeze(1), dim=-1)

    # Min-P 过滤（在 Top-K/Top-P 之前执行）
    if args.min_p is not None:
        max_probs = probs.max(dim=-1, keepdim=True).values
        threshold = max_probs * args.min_p.unsqueeze(1)
        probs[probs < threshold] = 0.0

    # 继续执行 Top-K / Top-P ...
```

### 第四步：在 API 层暴露参数

在 `python/minisgl/server/api_server.py` 的请求模型中添加 `min_p` 字段，并在构造 `SamplingParams` 时传入即可。

### 测试方法

```bash
curl http://localhost:1919/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_tokens": 64, "min_p": 0.1}'
```

对比 `min_p=0` 和 `min_p=0.1` 的输出多样性，验证过滤效果。

---

## 15.2 实例二：请求优先级调度

### 背景

当前 `PrefillManager`（文件 `python/minisgl/scheduler/prefill.py`）按照 FIFO 顺序处理 `pending_list` 中的请求。在生产环境中，常常需要支持请求优先级——让付费用户或低延迟需求的请求优先获得 GPU 资源。

### 第一步：定义优先级字段

在请求消息中添加优先级信息。首先修改 API 层的请求模型：

```python
# 文件: python/minisgl/server/api_server.py
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    priority: int = 0           # 新增：0 = 普通, 1 = 高优先级
```

优先级需要一路传递到 Scheduler 内部。在消息传递链路中，将 `priority` 附加到 `UserMsg` 对象上。

### 第二步：修改 PrefillManager

将 `pending_list` 从普通列表改为按优先级排序的结构：

```python
# 文件: python/minisgl/scheduler/prefill.py
import heapq

@dataclass
class PrefillManager:
    # ...
    pending_list: list = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        # 使用负优先级实现最大堆（高优先级先出队）
        heapq.heappush(self.pending_list, (-req.priority, id(req), req))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        # 按优先级从高到低取出请求
        reqs = []
        while self.pending_list and len(reqs) < budget:
            _, _, req = heapq.heappop(self.pending_list)
            # ... 调用 PrefillAdder.try_add_one() ...
```

### 第三步：考虑防饥饿机制

纯优先级调度可能导致低优先级请求永远得不到服务。一个简单的防饥饿策略是为每个请求记录等待时间，当等待超过阈值时自动提升优先级：

```python
def _boost_priority(self) -> None:
    now = time.monotonic()
    for item in self.pending_list:
        neg_prio, req_id, req = item
        if now - req.arrival_time > STARVATION_THRESHOLD:
            item[0] = -MAX_PRIORITY  # 提升到最高优先级
    heapq.heapify(self.pending_list)
```

### 测试方法

发送多个不同优先级的请求，观察它们的 TTFT 差异：

```python
import asyncio, aiohttp, time

async def test_priority():
    async with aiohttp.ClientSession() as session:
        # 先发送多个低优先级请求填满队列
        low_tasks = [session.post("http://localhost:1919/generate",
            json={"prompt": f"Low priority {i}", "max_tokens": 256, "priority": 0})
            for i in range(16)]
        # 再发送一个高优先级请求
        high_task = session.post("http://localhost:1919/generate",
            json={"prompt": "High priority", "max_tokens": 64, "priority": 1})
        # 高优先级请求的 TTFT 应显著更低
```

---

## 15.3 实例三：添加 Metrics 端点

### 背景

监控是生产部署的基础。我们为 Mini-SGLang 添加一个 `/metrics` 端点，暴露关键运行时指标。

### 第一步：定义指标收集器

创建一个简单的指标类：

```python
# 文件: python/minisgl/server/metrics.py（新建）
import time
from dataclasses import dataclass, field

@dataclass
class Metrics:
    total_requests: int = 0
    active_requests: int = 0
    total_tokens_generated: int = 0
    total_prefill_tokens: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.start_time

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_prefill_tokens": self.total_prefill_tokens,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "avg_tokens_per_second": round(
                self.total_tokens_generated / max(self.uptime_seconds, 1), 1
            ),
        }
```

### 第二步：在关键路径埋点

在 `Scheduler._process_last_data()` 中（文件 `python/minisgl/scheduler/scheduler.py`），每完成一个 Token 的 Decode 时递增计数器：

```python
# 文件: python/minisgl/scheduler/scheduler.py
def _process_last_data(self, batch, output):
    for req in completed_reqs:
        self.metrics.total_tokens_generated += req.output_len
    # ...
```

### 第三步：注册 API 端点

在 `python/minisgl/server/api_server.py` 中添加路由：

```python
# 文件: python/minisgl/server/api_server.py
from .metrics import Metrics

global_metrics = Metrics()

@app.get("/metrics")
async def get_metrics():
    return global_metrics.to_dict()
```

### 测试方法

```bash
# 发送一些请求后查看指标
curl http://localhost:1919/metrics
```

预期输出：

```json
{
  "total_requests": 42,
  "active_requests": 3,
  "total_tokens_generated": 5376,
  "total_prefill_tokens": 1280,
  "uptime_seconds": 120.5,
  "avg_tokens_per_second": 44.6
}
```

---

## 15.4 代码修改定位指南

下表汇总了常见扩展场景需要修改的文件：

| 扩展目标 | 需要修改的文件 |
|---|---|
| 新增 Sampling 方法 | `core.py`、`engine/sample.py`、`server/api_server.py` |
| 修改调度策略 | `scheduler/prefill.py`、`scheduler/decode.py` |
| 新增 API 端点 | `server/api_server.py` |
| 支持新模型架构 | `models/config.py`、新增 `models/<arch>.py` |
| 新增 Attention 后端 | `attention/base.py`、新增 `attention/<backend>.py` |
| 修改 KV Cache 策略 | `scheduler/cache.py`、`kvcache/base.py` |
| 添加新的环境变量 | `env.py` 中 `EnvClassSingleton` 类 |

---

## 15.5 测试策略

Mini-SGLang 的模块化设计使得各组件可以相对独立地进行测试。

### 单元测试

对纯计算逻辑（如 Sampling）编写单元测试：

```python
def test_min_p_sampling():
    logits = torch.randn(1, 32000)
    params = SamplingParams(temperature=1.0, min_p=0.1)
    # 验证过滤后的 Token 数量减少
    # 验证保留的 Token 概率 >= p_max * min_p
```

### 集成测试

使用 `--dummy-weight` 参数启动服务器进行端到端测试，这样无需下载真实模型权重：

```bash
python -m minisgl --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dummy-weight --port 9999
```

然后向 `/generate` 和 `/v1/chat/completions` 端点发送测试请求，验证响应格式正确、流式传输正常。

### 压力测试

使用并发工具（如 `wrk`、`locust` 或自定义脚本）验证高并发下的稳定性，确保调度器不会死锁、内存不会泄漏。

---

## 本章小结

1. 添加新的 Sampling 方法（如 Min-P）需要修改四处：`SamplingParams` 数据类、`Sampler.prepare()` 参数收集、`sample_impl()` 过滤逻辑、以及 API 层请求模型。
2. 实现请求优先级调度的核心改动在 `PrefillManager`，将 FIFO 队列替换为优先级队列，并配合防饥饿机制保证公平性。
3. 添加 Metrics 端点需要定义指标收集器、在 Scheduler 关键路径埋点、以及在 FastAPI 中注册新路由。
4. Mini-SGLang 的模块化架构使得扩展点清晰可循：Sampling 在 `engine/`、调度在 `scheduler/`、API 在 `server/`、模型在 `models/`。
5. 善用 `--dummy-weight` 参数可以在没有 GPU 模型权重的环境下进行功能开发和测试。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTIyMDY4MDhdfQ==
-->