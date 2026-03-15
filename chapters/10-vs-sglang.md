# 第 10 章 Mini vs SGLang：RadixAttention 的取舍

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away." — Antoine de Saint-Exupéry

SGLang 的核心创新之一是 RadixAttention——一种基于 Radix Tree 的 KV Cache 自动复用机制。Mini-SGLang 保留了这一思想的骨架，但大幅简化了实现。本章将深入对比两者，帮助读者理解"简化了什么"以及"为什么这样简化"。

---

## 10.1 RadixAttention：完整版 SGLang 的核心

### 什么是 RadixAttention

在 LLM 推理中，不同请求往往共享相同的前缀（system prompt、few-shot examples、共同的对话历史等）。传统做法是每个请求独立计算 KV Cache，造成大量重复计算和内存浪费。

RadixAttention 的核心思想是：**将所有请求的 KV Cache 组织为一棵 Radix Tree，相同前缀的 token 序列共享同一份物理内存。**

```
          root
         /    \
    [system    [system
     prompt     prompt
     A]         B]
    / \          |
  [Q1] [Q2]   [Q3]
```

在这棵树中，共享 "system prompt A" 的请求 Q1 和 Q2 只需存储一份 KV Cache。新请求到达时，系统沿树遍历，找到最长匹配前缀，只计算未匹配的部分。

### 完整版的关键机制

SGLang 的 RadixAttention 包含以下核心能力：

1. **自动前缀发现**：任意请求的 token 序列都会与 Radix Tree 做最长前缀匹配，无需手动标注哪些 token 是"可共享的"
2. **节点分裂（Split）**：当新序列与已有节点部分匹配时，自动将节点一分为二，保持 Radix Tree 的结构不变
3. **引用计数**：每个节点维护引用计数，追踪有多少活跃请求依赖这段 KV Cache
4. **LRU 驱逐**：引用计数归零的叶节点按最后使用时间排序，显存不足时优先驱逐
5. **跨请求复用**：同一棵树服务于所有并发请求，天然支持 chat 场景中多轮对话的前缀共享

---

## 10.2 Mini-SGLang 保留了什么

Mini-SGLang 的 `RadixPrefixCache`（位于 `kvcache/radix_cache.py`）确实实现了一棵 Radix Tree，保留了以下核心能力：

### 树遍历与前缀匹配

```python
# 文件: python/minisgl/kvcache/radix_cache.py
def _tree_walk(self, key):
    """沿 Radix Tree 遍历，返回最长匹配的节点和匹配长度"""
    # 从 root 开始，逐步比较 token 序列
    # 遇到不匹配时停止
```

`match_prefix` 方法通过 `_tree_walk` 实现前缀发现。当新请求到达时，系统不需要显式指定"这段是 system prompt"——只要 token 序列与树中已有路径匹配，就自动复用。

### 节点分裂

```python
# 文件: python/minisgl/kvcache/radix_cache.py
def split_at(self, node, split_pos):
    """在指定位置将节点一分为二
    保持 Radix Tree 的结构正确性"""
```

当新序列只匹配某个节点的前半部分时，`split_at` 将节点拆分，创建新的中间节点。这是 Radix Tree 区别于普通 Trie 的核心操作。

### LRU 驱逐

```python
# 文件: python/minisgl/kvcache/radix_cache.py
# 使用最小堆追踪叶节点时间戳
# 引用计数为 0 的叶节点是驱逐候选
```

Mini-SGLang 保留了基于时间戳的 LRU 驱逐。当 `CacheManager` 发现空闲页不足时，调用 `evict()` 方法，从最小堆中取出最老的叶节点进行回收。

### 引用计数与 Lock/Unlock

```python
# 文件: python/minisgl/kvcache/radix_cache.py
def lock_handle(self, handle):
    """增加引用计数，防止节点被驱逐"""

def unlock_handle(self, handle):
    """减少引用计数，允许节点被驱逐"""
```

Lock/Unlock 机制确保正在使用的 KV Cache 不会被驱逐。这直接对应 `BasePrefixCache` 中 `SizeInfo` 的 `evictable` 和 `protected` 两个计数器。

---

## 10.3 Mini-SGLang 简化了什么

### 简化一：单一 Key 生成策略

完整版 SGLang 支持多种 key 生成策略（per-token、per-page、per-image 等），以适配不同模态和不同粒度的缓存需求。Mini-SGLang 通过 `_get_key_fn` 工厂函数提供了 per-token 和 per-page 两种模式，但不支持多模态场景。

```python
# 文件: python/minisgl/kvcache/radix_cache.py
def _get_key_fn(self):
    """根据 page_size 选择 key 生成函数
    page_size=1: token 级别匹配
    page_size>1: 页级别匹配"""
```

### 简化二：无并发安全

完整版 SGLang 的 Radix Tree 需要处理多线程 / 多进程并发访问的安全性问题，包括读写锁、原子操作等。Mini-SGLang 假设单线程调度，不需要这些同步机制。

### 简化三：无 Speculative Decoding 集成

在生产版 SGLang 中，RadixAttention 需要与 Speculative Decoding 协作：投机生成的 token 可能被回滚，Radix Tree 需要支持高效的部分回退操作。Mini-SGLang 不涉及 Speculative Decoding，因此 Radix Tree 的操作更加简单。

### 简化四：无跨 Worker 缓存协调

完整版 SGLang 在 Tensor Parallel 模式下，多个 Worker 各自持有 KV Cache 的分片，但共享同一棵 Radix Tree 的逻辑结构。这要求缓存的分配和驱逐在所有 Worker 之间保持一致。Mini-SGLang 虽然支持基本的 Tensor Parallel 配置，但缓存管理本身不涉及跨 Worker 协调。

---

## 10.4 完整对比

| 特性 | SGLang (完整版) | Mini-SGLang |
|-----|----------------|-------------|
| Radix Tree 结构 | 完整实现 | 完整实现 |
| 前缀匹配 (tree walk) | 支持 | 支持 |
| 节点分裂 (split) | 支持 | 支持 |
| LRU 驱逐 | 支持，多策略 | 支持，单策略 |
| 引用计数 | 支持 | 支持 |
| Lock/Unlock | 支持 | 支持 |
| 多模态 Key | 支持 | 不支持 |
| 并发安全 | 支持 | 不需要 |
| Speculative Decoding | 集成 | 不涉及 |
| 跨 Worker 协调 | 支持 | 不涉及 |
| Chunked Prefill 集成 | 深度集成 | 基本集成 |

---

## 10.5 为什么这样取舍

Mini-SGLang 的取舍遵循一个清晰的原则：**保留算法骨架，去除工程复杂度。**

Radix Tree 的核心算法——tree walk、split、LRU eviction、引用计数——在 Mini-SGLang 中完整保留。读者阅读这些代码，能够理解 RadixAttention "为什么 work"：为什么 Radix Tree 比 Hash Map 更适合前缀匹配，为什么 LRU 比 FIFO 更适合 KV Cache 驱逐，为什么引用计数能防止使用中的缓存被误删。

被去掉的是工程层面的"防御性"代码：并发锁、多模态适配、跨 Worker 同步。这些机制解决的是"如何在生产环境中安全运行"的问题，而非"算法为什么正确"的问题。对于学习者而言，它们是噪音而非信号。

### 一个具体的例子

考虑 `insert_prefix` 操作。在 Mini-SGLang 中，这是一个清晰的三步过程：

1. 沿树遍历找到插入位置
2. 如果需要，分裂已有节点
3. 创建新的叶节点

在完整版 SGLang 中，同样的操作还需要：

- 获取写锁，防止并发修改
- 通知所有 TP Worker 更新本地的 KV Cache 索引
- 处理可能的内存分配失败（OOM 回退）
- 更新多级统计计数器

核心算法相同，但代码量可能相差 3-5 倍。Mini-SGLang 让读者用 1/5 的代码理解 5/5 的算法思想。

---

## 本章小结

1. **RadixAttention** 是 SGLang 的核心创新，通过 Radix Tree 实现 KV Cache 的自动前缀复用，避免重复计算。
2. **Mini-SGLang 保留了 Radix Tree 的完整算法骨架**：tree walk、节点分裂、LRU 驱逐、引用计数与 Lock/Unlock 机制。
3. **被简化的主要是工程层面的复杂度**：并发安全、多模态适配、Speculative Decoding 集成、跨 Worker 缓存协调。
4. **取舍原则清晰**：保留"算法为什么正确"，去除"如何在生产中安全运行"，让学习者聚焦核心思想。
5. 读者理解了 Mini-SGLang 的 RadixPrefixCache 后，阅读完整版 SGLang 的同名模块时会发现：核心逻辑一致，区别在于更多的边界处理和系统集成。
