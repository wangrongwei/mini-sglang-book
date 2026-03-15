# 附录 C：名词解释

> "The beginning of wisdom is the definition of terms." —— Socrates

本附录按字母顺序收录全书涉及的核心术语，提供简明中文定义和相关章节引用，便于读者在阅读过程中随时查阅。

---

## A

**Attention（注意力机制）**
Transformer 的核心计算模块。给定 Query、Key、Value 三组向量，通过 $\text{Softmax}(QK^T/\sqrt{d})V$ 计算加权输出。Mini-SGLang 中由 `attention/base.py:BaseAttnBackend.forward()` 抽象，支持多种后端实现。
→ 相关章节：第 7 章

**Autoregressive（自回归）**
语言模型的生成方式：每一步根据已有 Token 预测下一个 Token，新 Token 拼接到序列末尾后继续预测。这意味着生成 $N$ 个 Token 需要 $N$ 次前向推理。
→ 相关章节：第 1 章、第 5 章

---

## B

**Batch（批次）**
将多个请求的 Token 合并为一次 GPU 计算，提高硬件利用率。Mini-SGLang 中由 `core.py:Batch` 表示，包含 `input_ids`、`positions`、`out_loc` 等张量。
→ 相关章节：第 3 章

**BFloat16（BF16）**
一种 16 位浮点格式，保留与 Float32 相同的指数范围，牺牲尾数精度。在大模型推理中被广泛使用，兼顾速度与数值稳定性。通过 `--dtype bfloat16` 启用。
→ 相关章节：第 13 章

---

## C

**Chunked Prefill（分块预填充）**
将长序列的 Prefill 拆分为多个小块逐步执行，避免单次 Prefill 占用过多 GPU 资源而阻塞 Decode 请求。Mini-SGLang 通过 `--max-prefill-length` 参数控制分块大小，`scheduler/prefill.py:ChunkedReq` 管理分块状态。
→ 相关章节：第 5 章

**Continuous Batching（连续批处理）**
与静态 Batching 不同，Continuous Batching 允许请求在任意时刻加入或离开 Batch，最大化 GPU 利用率。调度器在每步 Decode 后动态调整 Batch 组成。这是 Mini-SGLang 的核心调度理念。
→ 相关章节：第 4 章、第 5 章

**CUDA Graph**
NVIDIA 提供的机制，将一系列 CUDA Kernel 调用录制为 Graph，后续执行时直接回放，消除 CPU-GPU 之间的 Kernel Launch 开销。Mini-SGLang 在 `engine/graph.py:GraphRunner` 中实现，对小 Batch Decode 阶段加速效果显著。
→ 相关章节：第 8 章

---

## D

**Decode（解码阶段）**
自回归生成中的逐 Token 生成阶段。每步仅计算一个新 Token 的 Attention，但需要访问所有历史 KV Cache。Compute Bound 程度低，主要受 Memory Bandwidth 限制。由 `scheduler/decode.py:DecodeManager` 管理。
→ 相关章节：第 5 章、第 6 章

**Detokenizer（反分词器）**
将模型输出的 Token ID 序列转换回人类可读文本的组件。Mini-SGLang 中作为独立进程运行，通过 ZMQ 与 Scheduler 通信。
→ 相关章节：第 12 章、第 13 章

---

## E

**Eviction（驱逐）**
当 KV Cache 空间不足时，按策略（如 LRU）回收已缓存的页面以腾出空间。`scheduler/cache.py:CacheManager.evict()` 实现此功能，配合前缀缓存使用。
→ 相关章节：第 9 章

---

## F

**FlashAttention**
一种 IO-aware 的 Attention 算法，通过分块计算和在线 Softmax 技巧将 Attention 的显存占用从 $O(N^2)$ 降至 $O(N)$，同时利用 GPU SRAM 加速计算。Mini-SGLang 通过 `attention/fa.py` 集成。
→ 相关章节：第 7 章

**FlashInfer**
一个高性能 Attention 和 Sampling 库，提供 PagedAttention、Top-K/Top-P Sampling 等 GPU Kernel 实现。Mini-SGLang 在 `engine/sample.py:sample_impl()` 中使用其采样功能。
→ 相关章节：第 7 章、第 10 章

**Forward Pass（前向传播）**
模型推理的核心计算过程：输入 Token 经过 Embedding、多层 Transformer Block、最终 LM Head，输出 Logits 向量。`engine/engine.py:Engine.forward_batch()` 封装此过程。
→ 相关章节：第 8 章

---

## G

**Greedy Decoding（贪心解码）**
每步选择概率最高的 Token 作为输出，即 `argmax(logits)`。确定性强但多样性差。当 `temperature=0` 或 `top_k=1` 时触发。
→ 相关章节：第 10 章

---

## H

**HybridBackend（混合后端）**
Mini-SGLang 的 Attention 后端组合模式，允许 Prefill 和 Decode 阶段使用不同的 Attention 实现。由 `attention/base.py:HybridBackend` 实现。
→ 相关章节：第 7 章

---

## K

**KV Cache（键值缓存）**
存储 Attention 层已计算的 Key 和 Value 向量，避免重复计算。KV Cache 的大小随序列长度线性增长，是 GPU 显存的最大消费者之一。`kvcache/base.py:BaseKVCachePool` 管理物理存储。
→ 相关章节：第 6 章

---

## L

**Logits**
模型最后一层输出的未归一化分数向量，维度等于词表大小。经过 Softmax 转换为概率分布后用于采样。
→ 相关章节：第 10 章

---

## M

**MoE（Mixture of Experts，混合专家）**
一种模型架构，每层包含多个 Expert（专家网络），通过 Gate 网络动态选择部分 Expert 参与计算。Qwen3 等模型使用此架构。Mini-SGLang 的 `models/config.py:ModelConfig.is_moe` 属性用于检测。
→ 相关章节：第 11 章

---

## O

**Overlap Scheduling（重叠调度）**
将上一轮推理结果的处理（CPU 工作）与当前轮推理的 GPU 计算重叠执行，隐藏 CPU 延迟。`scheduler/scheduler.py:overlap_loop()` 实现此策略。
→ 相关章节：第 4 章

---

## P

**Page（页面）**
KV Cache 的最小分配单位。每页存储固定数量的 Token 的 KV 向量（由 `--page-size` 参数控制）。页式管理避免了预分配最大长度的显存浪费。
→ 相关章节：第 6 章

**Page Table（页表）**
记录每个请求的逻辑 Token 位置到物理 KV Cache 页面的映射关系。`scheduler/cache.py:_write_page_table()` 负责填充。
→ 相关章节：第 6 章

**PagedAttention**
将 KV Cache 按页管理的 Attention 实现，灵感来自操作系统的虚拟内存。允许非连续的物理存储，极大提高了显存利用率。
→ 相关章节：第 6 章、第 7 章

**Prefill（预填充阶段）**
处理用户输入 Prompt 的阶段，一次性计算所有输入 Token 的 Attention 并填充 KV Cache。计算量大但高度并行。由 `scheduler/prefill.py:PrefillManager` 管理。
→ 相关章节：第 5 章

**Prefix Caching（前缀缓存）**
缓存相同前缀的 KV Cache 以供后续请求复用，避免重复 Prefill 计算。Mini-SGLang 的 `CacheManager.match_prefix()` 和 `insert_prefix()` 实现此功能。
→ 相关章节：第 9 章

---

## R

**RadixAttention（基数树注意力）**
SGLang 提出的前缀缓存方案，使用 Radix Tree（基数树）数据结构高效管理共享前缀。Mini-SGLang 实现了其简化版本。
→ 相关章节：第 9 章

**Req（请求）**
推理系统中的基本工作单元，对应一次用户生成请求。`core.py:Req` 类追踪输入 Token、缓存状态、输出长度等信息。
→ 相关章节：第 3 章

**RoPE（Rotary Position Embedding，旋转位置编码）**
一种相对位置编码方法，通过旋转变换将位置信息注入 Query 和 Key 向量。Llama、Qwen 等模型采用此方案。配置由 `models/config.py:RotaryConfig` 管理。
→ 相关章节：第 11 章

---

## S

**SamplingParams（采样参数）**
控制 Token 采样行为的参数集合，包括 `temperature`、`top_k`、`top_p`、`max_new_tokens` 等。由 `core.py:SamplingParams` 定义。
→ 相关章节：第 10 章

**Scheduler（调度器）**
推理系统的核心编排器，负责管理请求队列、分配 GPU 资源、组装 Batch、协调 Prefill 与 Decode 阶段。`scheduler/scheduler.py:Scheduler` 实现。
→ 相关章节：第 4 章

**Sequence（序列）**
一串有序的 Token，可以是用户输入（Prompt）或模型输出。序列长度决定了 KV Cache 占用和 Attention 计算量。
→ 相关章节：第 1 章

**SSE（Server-Sent Events）**
一种 HTTP 流式传输协议，服务器可以持续向客户端推送数据。Mini-SGLang 的 `/generate` 端点使用 SSE 逐 Token 返回结果。
→ 相关章节：第 12 章

---

## T

**Temperature（温度）**
采样温度参数，控制输出的随机性。$T > 1$ 使分布更平坦（更随机），$T < 1$ 使分布更尖锐（更确定），$T = 0$ 等价于 Greedy Decoding。在 `sample_impl()` 中通过 $\text{logits} / T$ 实现。
→ 相关章节：第 10 章

**Tensor Parallelism（张量并行）**
将模型的权重矩阵沿特定维度切分到多块 GPU 上并行计算。Mini-SGLang 通过 `--tensor-parallel-size` 参数配置，每个 TP rank 运行一个独立的 Scheduler 进程。
→ 相关章节：第 8 章、第 13 章

**Token**
文本的最小处理单位。Tokenizer 将文本切分为 Token 序列，每个 Token 对应词表中的一个整数 ID。模型以 Token 为单位进行输入和输出。
→ 相关章节：第 1 章

**Tokenizer（分词器）**
将文本转换为 Token ID 序列的组件。Mini-SGLang 使用 HuggingFace Transformers 的 Tokenizer，在独立进程中运行。
→ 相关章节：第 12 章

**Top-K Sampling**
只保留概率最高的 $K$ 个 Token，将其余 Token 的概率置零后重新归一化采样。$K=1$ 等价于 Greedy Decoding。
→ 相关章节：第 10 章

**Top-P Sampling（Nucleus Sampling）**
按概率从高到低累积，只保留累积概率达到 $P$ 的最小 Token 集合。相比 Top-K 更具自适应性——高确信时候选少，低确信时候选多。
→ 相关章节：第 10 章

---

## Z

**ZMQ（ZeroMQ）**
一个高性能异步消息库，Mini-SGLang 用它在 API Server、Tokenizer、Scheduler、Detokenizer 进程之间传递消息。`server/api_server.py:FrontendManager` 管理 ZMQ 连接。
→ 相关章节：第 12 章

---

## 本附录小结

1. 本术语表涵盖了 LLM 推理系统中从基础概念（Token、Attention）到系统优化（CUDA Graph、Overlap Scheduling）的 40 余个核心术语。
2. 每个术语都标注了在 Mini-SGLang 源码中的对应实现位置，便于从概念回溯到代码。
3. 建议初次阅读本书时先浏览本附录建立概念框架，在深入各章时再按需查阅具体定义。
4. 术语间存在大量关联——例如 PagedAttention 依赖 Page 和 Page Table 概念，Continuous Batching 与 Scheduler 和 Decode 紧密相关——建议交叉参考理解。
