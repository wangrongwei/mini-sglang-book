---
layout: home

hero:
  name: "Mini-SGLang 源码解析"
  text: "500 行代码看懂推理引擎核心"
  tagline: SGLang 最小可读实现深度剖析——从 Request 到 Token 输出，逐行拆解 Continuous Batching、KV Cache 与调度器的极简实现
  actions:
    - theme: brand
      text: 开始阅读
      link: /chapters/01-positioning
    - theme: alt
      text: 查看目录
      link: /contents
    - theme: alt
      text: GitHub
      link: https://github.com/sgl-project/mini-sglang

features:
  - icon:
      src: /icons/engine.svg
    title: 推理引擎核心
    details: 深入 Prefill-Decode 两阶段管线，解析 token 生成、attention 计算与 logits 采样的完整实现，用最少代码理解 LLM 推理本质。

  - icon:
      src: /icons/cache.svg
    title: KV Cache 管理
    details: 剖析张量预分配、slot 管理与内存回收机制，理解推理引擎中最关键的性能优化点——如何用简化实现还原生产级 KV Cache 设计。

  - icon:
      src: /icons/scheduler.svg
    title: 调度与批处理
    details: 拆解 Continuous Batching 的最小实现，理解等待队列、运行队列、内存感知调度如何在 500 行代码内完成动态批处理。

  - icon:
      src: /icons/compare.svg
    title: 与生产实现对比
    details: 逐一对比 Mini-SGLang 与 SGLang/vLLM 的设计取舍，理解 RadixAttention、PagedAttention 等生产级特性的演进路径。
---
