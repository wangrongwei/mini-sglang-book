# 第 1 章 Mini-SGLang 的定位

> "The best way to understand a system is to build a minimal version of it."
> —— 系统设计的古老智慧

大语言模型（LLM）推理引擎是当下 AI 基础设施中最核心的组件之一。SGLang、vLLM、TensorRT-LLM 等生产级系统动辄数十万行代码，涵盖了分布式通信、内存管理、算子融合、多后端适配等复杂工程。对于希望深入理解推理引擎内部机制的开发者而言，直接阅读生产代码的门槛极高。Mini-SGLang 正是为解决这一问题而生——它是 SGLang 团队官方维护的**最小可读实现**，用约 5000 行 Python 代码还原了推理引擎的核心概念。

---

## 1.1 为什么需要 Mini-SGLang

生产级推理引擎面临的典型阅读障碍包括：

| 障碍类型 | 生产代码的表现 | Mini-SGLang 的处理 |
|---------|--------------|-------------------|
| 代码量 | SGLang 主仓库 10 万+ 行 | 约 5000 行 Python |
| 平台抽象 | 多后端（CUDA/ROCm/CPU）适配层 | 仅保留 CUDA 路径 |
| 性能优化 | 大量 C++/CUDA kernel、内存池 | 用 PyTorch 原生操作 + 少量自定义 kernel |
| 分布式 | 完整的 TP/PP/EP 支持 | 仅保留基础 Tensor Parallelism |
| 功能覆盖 | LoRA、Speculative Decoding、Vision 等 | 聚焦文本生成核心路径 |

Mini-SGLang 的设计哲学可以概括为一句话：**保留架构骨架，删除工程复杂度**。它不是一个玩具——它实际可以加载 Llama、Qwen 等主流模型并以有竞争力的性能运行推理，同时保持每个模块都可以在一个屏幕内读完。

---

## 1.2 目标读者

Mini-SGLang 源码解析适合以下读者：

1. **推理引擎开发者**：正在或计划参与 SGLang/vLLM 等项目开发，需要快速建立全局认知
2. **ML 系统研究者**：研究 Continuous Batching、KV Cache 管理、调度策略等方向，需要一个可修改的实验平台
3. **高级 LLM 用户**：使用推理引擎部署模型，希望理解 `max_tokens`、`temperature` 等参数背后的实现逻辑

你需要具备的前置知识：

- 熟悉 Python 和 PyTorch 基础操作
- 了解 Transformer 架构（Self-Attention、KV Cache 的基本概念）
- 对 GPU 编程有初步认知（知道 CUDA stream、device/host 的区别）

---

## 1.3 Mini-SGLang 覆盖了什么

尽管只有约 5000 行代码，Mini-SGLang 覆盖了推理引擎的**完整关键路径**：

```
用户请求 → Tokenization → 调度器 → Prefill → KV Cache 写入
    → Decode 循环 → Sampling → Detokenization → 流式响应
```

具体包含的核心机制：

- **Continuous Batching**：动态将新请求插入正在运行的 batch，而非等待整个 batch 完成
- **Paged KV Cache**：基于页式管理的 KV Cache 分配与回收
- **Radix Prefix Cache**：基于 Radix Tree 的前缀复用，避免重复计算共享 prompt
- **Chunked Prefill**：将长 prompt 的 prefill 分块执行，控制单次计算的内存峰值
- **Overlap Scheduling**：CPU 调度与 GPU 计算重叠执行
- **CUDA Graph Capture**：对 decode 阶段的固定形状计算进行图捕获加速
- **Tensor Parallelism**：跨多 GPU 的模型并行

---

## 1.4 简化了什么，为什么这样简化

理解 Mini-SGLang **不做什么**与理解它做什么同样重要：

**不支持 Pipeline Parallelism（PP）**：PP 引入跨设备的微批次流水线调度，显著增加调度器复杂度。Mini-SGLang 仅保留 TP，因为 TP 是理解模型并行的最小必要概念。

**不支持 LoRA / Speculative Decoding**：这些是正交的功能特性，不影响对核心推理管线的理解。

**不做极致的内存优化**：生产级 SGLang 中的内存池、预分配策略、zero-copy 传输等被简化为 PyTorch 标准张量操作。

**模型支持有限**：仅实现 Llama、Qwen2、Qwen3、Mistral 等架构（见 `python/minisgl/models/` 目录），不支持视觉模型或 Encoder-Decoder 架构。

每一项简化都遵循同一原则：**如果删除它不影响读者理解"一个 token 是怎么从输入变成输出的"，就删除它。**

---

## 1.5 本书的阅读路径

本书按照从外到内、从静态到动态的顺序组织：

```
第一部分：定位与全览          ← 你在这里
    ↓
第二部分：核心数据结构        ← Req、Batch、KV Cache
    ↓
第三部分：Prefill 与 Decode   ← 两阶段执行流程
    ↓
第四部分：调度器              ← Continuous Batching 实现
    ↓
第五部分：与生产实现对比      ← Mini vs SGLang vs vLLM
    ↓
第六部分：动手实验            ← 运行与扩展
```

每一章都会标注对应的源文件路径（如 `# 文件: python/minisgl/core.py`），读者可以随时对照源码阅读。

---

## 1.6 如何使用本书

推荐的阅读方式：

1. **先通读第 2 章**（代码全览），建立模块间的依赖关系认知
2. **按章节顺序阅读**第 3-9 章，每章配合源码
3. **跳到第 13 章**实际运行 Mini-SGLang，用 debugger 单步跟踪关键路径
4. **回到第 10-12 章**，在理解 Mini 实现的基础上对比生产代码

本书的代码引用始终指向 Mini-SGLang 仓库（`https://github.com/sgl-project/mini-sglang`）的 `python/minisgl/` 目录。所有代码块上方会以注释标注文件路径，例如：

```python
# 文件: python/minisgl/core.py
@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
```

---

## 本章小结

1. Mini-SGLang 是 SGLang 团队官方维护的最小可读推理引擎实现，约 5000 行 Python 代码
2. 它的设计哲学是"保留架构骨架，删除工程复杂度"——覆盖从请求接收到 token 输出的完整关键路径
3. 简化的判断标准是：是否影响读者理解"一个 token 是怎么从输入变成输出的"
4. 本书按"定位 → 数据结构 → 执行流程 → 调度器 → 生产对比 → 动手实验"的路径组织，每章对应具体源文件
5. 目标读者是希望深入理解推理引擎内部机制的开发者、研究者和高级用户
