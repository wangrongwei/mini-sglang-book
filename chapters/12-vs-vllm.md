# 第 12 章 Mini vs vLLM：BlockManager 的取舍

> "All problems in computer science can be solved by another level of indirection." — David Wheeler

vLLM 凭借 PagedAttention 开创了 LLM 推理的分页内存管理范式。Mini-SGLang 采用了相似的分页思想，但在实现上走了一条更简洁的路。本章将深入对比 vLLM 的 BlockManager 与 Mini-SGLang 的 page_table 方案，帮助读者理解两种设计哲学的异同。

---

## 12.1 vLLM 的 PagedAttention 架构

### 核心思想

vLLM 的 PagedAttention 受操作系统虚拟内存启发，将 KV Cache 分割为固定大小的 Block，通过 Block Table 实现逻辑地址到物理地址的映射。其核心抽象包括：

```
Sequence → Logical Blocks → Block Table → Physical Blocks → GPU Memory
```

每个 Sequence 持有一组 Logical Block，每个 Logical Block 通过 Block Table 映射到一个 Physical Block。Physical Block 是 GPU 显存中实际存储 KV 对的连续区域。

### BlockManager 的关键组件

vLLM 的内存管理由多层抽象组成：

| 组件 | 职责 |
|-----|------|
| **BlockSpaceManager** | 顶层管理器，决定是否有足够空间接纳新请求 |
| **BlockAllocator** | 管理物理 Block 的分配与释放 |
| **Block Table** | 维护逻辑 Block → 物理 Block 的映射关系 |
| **Sequence** | 单个生成序列，持有逻辑 Block 列表 |
| **SequenceGroup** | 一组共享 prompt 的序列（用于 beam search） |

### Copy-on-Write (CoW)

vLLM 最精妙的设计之一是 Copy-on-Write。当多个 Sequence 共享同一个 Physical Block 时（例如 beam search 中的不同分支），Block 的引用计数大于 1。当某个 Sequence 需要修改这个 Block 时，系统才复制一份新的 Physical Block：

```
Seq A ─┐                    Seq A ──→ Block X (copy)
       ├──→ Block X    →
Seq B ─┘                    Seq B ──→ Block X (original)
```

这种机制在 beam search 和 parallel sampling 场景下显著节省内存——共享 prompt 部分的 KV Cache 只存储一份，直到各分支产生不同的 token 才分裂。

### Prefix Caching

vLLM 后续版本也引入了 prefix caching，通过对 Block 内容计算 hash 来识别可复用的前缀。这与 SGLang 的 Radix Tree 方案不同——vLLM 用 hash-based 方案，SGLang 用 tree-based 方案。

---

## 12.2 Mini-SGLang 的 page_table 方案

### 简洁的映射

Mini-SGLang 使用一个全局的 `page_table` 张量，直接维护请求到物理页的映射：

```python
# 文件: python/minisgl/engine/engine.py
# 创建 page_table 张量
# 维度: [max_requests, max_pages_per_request]
# 值: 物理页编号
```

这个 `page_table` 存储在 `Context` 对象中，被 Scheduler 和 Attention Backend 共享访问：

```python
# 文件: python/minisgl/core.py
@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor  # 全局页表
    kv_cache: BaseKVCachePool  # KV Cache 存储池
```

### 分配流程

`CacheManager.allocate_paged` 在分配时直接操作 `page_table` 张量：

```python
# 文件: python/minisgl/scheduler/cache.py
def allocate_paged(self, reqs):
    # 对每个请求:
    #   1. 从 free_slots 取出所需数量的页
    #   2. 将物理页编号写入 page_table[req_idx, :]
    #   3. 更新 req 的缓存状态
```

没有 Logical Block、没有 BlockAllocator、没有 BlockSpaceManager——一个张量搞定所有映射。

---

## 12.3 逐项对比

### 抽象层次

```
vLLM:    Sequence → LogicalBlock → BlockTable → PhysicalBlock → GPU Memory
                                                    ↑
                                              BlockAllocator

Mini:    Req → page_table[req_idx] → Physical Page → GPU Memory
                                          ↑
                                      free_slots (set)
```

vLLM 的多层抽象为 beam search 和 prefix sharing 等高级特性提供了基础设施。Mini-SGLang 的两层映射虽然不支持这些特性，但足以演示分页内存管理的核心原理。

### Copy-on-Write

| 方面 | vLLM | Mini-SGLang |
|-----|------|-------------|
| 是否支持 | 支持 | 不支持 |
| 实现方式 | Block 引用计数 + 写时复制 | 无 |
| 适用场景 | beam search, parallel sampling | 仅 greedy / sampling |
| 内存节省 | beam search 可节省 60%+ | 无此优化 |

Mini-SGLang 不支持 CoW，因为它的目标场景是单序列生成（greedy 或 top-k/top-p sampling），不涉及同一 prompt 的多序列并行生成。在教学中去掉 CoW 可以让读者专注于理解分页本身，而不被引用计数和块复制的细节分散注意力。

### Block 管理粒度

vLLM 的 BlockAllocator 支持：

- **分层分配**：区分 GPU Block 和 CPU Block，支持 swap（将 Block 从 GPU 换出到 CPU）
- **碎片整理**：定期重新组织 Block，减少内存碎片
- **Block 共享**：多个 Sequence 共享同一 Block

Mini-SGLang 的 `free_slots` 集合是一个扁平的页池，不区分存储层级，不支持 swap，也不做碎片整理。

### Prefix Caching 方式

| 方面 | vLLM | Mini-SGLang |
|-----|------|-------------|
| 方案 | Hash-based Block 匹配 | Radix Tree 前缀匹配 |
| 匹配粒度 | Block 级别 | Token/Page 级别 |
| 优点 | O(1) 查找，实现简单 | 天然支持任意长度前缀 |
| 缺点 | Hash 冲突，Block 边界敏感 | 树维护开销 |

这是一个有趣的设计差异：vLLM 和 SGLang 对同一个问题（前缀复用）选择了不同的数据结构，各有优劣。Mini-SGLang 继承了 SGLang 的 Radix Tree 路线。

---

## 12.4 简化的代价

坦诚地说，Mini-SGLang 的简化是有代价的：

### 1. 无法支持 Beam Search

没有 CoW，beam search 中每个候选序列都需要独立的 KV Cache 副本，内存开销线性增长。对于需要 beam search 的应用（如翻译、摘要），这是一个实质性限制。

### 2. 无法跨层级调度内存

vLLM 的 swap 机制允许在 GPU 显存不足时将 Block 换出到 CPU 内存，而非直接驱逐。这意味着 vLLM 可以在 GPU 满时"暂停"低优先级请求，稍后恢复，而不需要重新计算。Mini-SGLang 没有这个能力，驱逐就意味着丢失，下次访问需要重算。

### 3. 内存碎片

vLLM 通过 BlockAllocator 的内部管理可以缓解碎片问题。Mini-SGLang 的简单 set 管理方式在长时间运行后可能产生碎片，虽然在教学场景中这通常不是问题。

---

## 12.5 设计哲学的差异

从更高的视角来看，vLLM 和 Mini-SGLang 代表了两种不同的设计哲学：

**vLLM 的哲学：从操作系统借鉴。** Block 的概念直接对应 OS 中的 Page Frame，BlockTable 对应 Page Table，CoW 对应 fork() 的写时复制语义，swap 对应虚拟内存的页面换出。vLLM 把数十年操作系统研究的成果搬到了 GPU 内存管理中。

**Mini-SGLang 的哲学：最小可教学实现。** 保留分页的核心概念（固定大小的页、逻辑到物理的映射），去掉不影响核心理解的高级特性。让读者在 500 行代码而非 5000 行中理解"为什么 LLM 推理需要分页"。

两者并不矛盾。Mini-SGLang 可以作为理解 vLLM 的跳板——先在简化版中建立直觉，再在完整版中看到每个抽象层的存在理由。

---

## 本章小结

1. **vLLM 的 PagedAttention** 采用多层抽象（Sequence → LogicalBlock → BlockTable → PhysicalBlock），支持 Copy-on-Write、swap、prefix caching 等高级特性。
2. **Mini-SGLang 用单一 page_table 张量**实现逻辑到物理的映射，去掉了 BlockAllocator、SequenceGroup 等中间抽象。
3. **Copy-on-Write 是最大的差异**：vLLM 通过 CoW 支持 beam search 和 parallel sampling 的内存共享，Mini-SGLang 不涉及此场景。
4. **Prefix Caching 路线不同**：vLLM 用 hash-based Block 匹配，Mini-SGLang 继承 SGLang 的 Radix Tree 方案。
5. **简化的代价**包括无法支持 beam search、无 GPU-CPU swap、潜在的内存碎片。
6. **两者互补**：Mini-SGLang 帮助读者建立分页内存管理的核心直觉，vLLM 展示了生产级系统如何在此基础上构建完整的内存管理栈。
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEwOTA3NDg1Ml19
-->