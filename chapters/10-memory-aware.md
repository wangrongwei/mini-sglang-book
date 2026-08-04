# 第 10 章 内存感知调度

> "The art of memory management is deciding what to forget." — Anonymous

GPU 显存是推理服务中最稀缺的资源。一个设计良好的调度器不仅要决定"下一个处理谁"，更要回答"还能不能再接一个请求"。Mini-SGLang 通过 `CacheManager` 与 `PrefillAdder` 的协作，实现了一套轻量但完整的内存感知调度机制。本章将沿着一个请求从提交到被接纳（或被拒绝）的路径，逐层剖析内存预算的计算、页分配与驱逐策略。

---

## 10.1 内存预算：从物理显存到逻辑页

在 Engine 初始化阶段，系统首先计算可用于 KV Cache 的总显存：

```python
# 文件: python/minisgl/engine/engine.py
cache_per_page = 2 * config.model_config.head_dim * div_even(
    config.model_config.num_kv_heads, tp_size
) * config.model_config.num_layers * page_size
```

这里的 `cache_per_page` 表示单个逻辑页在所有层的 KV 缓存所占的字节数。因子 `2` 对应 Key 和 Value 两个张量；`div_even` 在 Tensor Parallel 下均分 KV Head 数。系统用剩余显存除以 `cache_per_page`，得到总页数 `num_pages`，这就是全局的内存预算上限。

### page_size 与内存粒度

`page_size` 是连接逻辑调度与物理内存的关键参数：

| page_size | 每页 token 数 | 内存粒度 | 碎片风险 | 前缀匹配粒度 |
|-----------|-------------|---------|---------|------------|
| 1         | 1           | 最细     | 最低     | token 级    |
| 16        | 16          | 中等     | 中等     | 16-token 块 |
| 128       | 128         | 最粗     | 最高     | 128-token 块 |

Mini-SGLang 默认使用较小的 `page_size`，以在教学场景中展示最清晰的分配语义。生产系统通常选取 16 或更大的值，在管理开销与碎片之间取得平衡。

---

## 10.2 CacheManager：页的分配与回收

`CacheManager` 是内存感知调度的核心组件，维护三类状态：

1. **free_slots**：空闲物理页的集合
2. **prefix cache**：缓存了前缀 token 的页（可被驱逐）
3. **page_table**：从逻辑页号到物理页号的映射张量

### 分配流程：allocate_paged

当 Scheduler 准备一个 batch 时，`CacheManager.allocate_paged` 负责为 batch 中每个请求分配所需的页：

```python
# 文件: python/minisgl/scheduler/cache.py
def allocate_paged(self, reqs):
    # 1. 计算每个请求需要的新页数
    # 2. 检查总需求是否超过 free_slots
    # 3. 如果不足，触发驱逐
    # 4. 从 free_slots 中取出页，写入 page_table
```

关键在于第 2 步的检查——这是"内存感知"的直接体现。如果空闲页不够，系统不会直接拒绝请求，而是先尝试驱逐前缀缓存中的可回收页。

### 驱逐策略：evict

当 `free_slots` 不足以满足新分配需求时，`CacheManager` 调用底层 `BasePrefixCache.evict()` 方法：

```python
# 文件: python/minisgl/kvcache/base.py
def evict(self, num_tokens: int) -> int:
    """驱逐至少 num_tokens 个 token 的缓存
    注意：实际驱逐量可能大于请求量"""
```

Mini-SGLang 的 `RadixPrefixCache` 使用 LRU（Least Recently Used）策略，通过最小堆跟踪叶节点的最后访问时间戳，优先驱逐最久未使用的前缀。被驱逐的页回到 `free_slots`，供新请求使用。

---

## 10.3 Scheduler 的准入决策

调度器在将请求加入 prefill batch 之前，通过 `PrefillAdder` 进行逐个准入检查：

```python
# 文件: python/minisgl/scheduler/prefill.py
class PrefillAdder:
    def _try_allocate_one(self, req):
        # 1. 尝试前缀匹配 (cache.match_req)
        # 2. 计算还需要多少新页
        # 3. 检查 cache 是否有足够空间
        # 4. 分配成功 → 加入 batch；失败 → 请求留在等待队列
```

这里的决策逻辑是保守的：如果分配失败，请求不会被丢弃，而是继续留在 `PrefillManager` 的 pending 队列中，等待下一次调度循环。这种"延迟而非拒绝"的策略确保了请求不会因为瞬时的内存压力而丢失。

### Token 预算与分块

除了页级别的内存检查，`PrefillAdder` 还维护一个 token 预算（token budget），限制单次 prefill batch 的总 token 数。当一个请求的 `extend_len`（需要新计算的 token 数）超过剩余预算时，系统会将请求切分为 `ChunkedReq`：

```python
# 文件: python/minisgl/scheduler/prefill.py
class ChunkedReq:
    """阻止采样和解码的分块请求
    大请求被拆分到多个 batch 中处理"""
```

分块机制是内存感知的另一个维度——它不仅控制"用多少页"，还控制"一次算多少 token"，防止单个大请求独占计算资源。

---

## 10.4 Decode 阶段的内存追踪

进入 decode 阶段后，`DecodeManager` 通过 `inflight_tokens` 属性持续追踪在途内存占用：

```python
# 文件: python/minisgl/scheduler/decode.py
@property
def inflight_tokens(self):
    """计算所有运行中请求的已预留 + 剩余 token 总量"""
```

这个指标被 Scheduler 用来动态调节 prefill 的准入速率——如果 decode 阶段占用的内存已经很高，新的 prefill 请求会被延迟接纳，避免 OOM。

---

## 10.5 Lazy Free：延迟释放的艺术

`CacheManager` 提供了一个 `lazy_free_region` 上下文管理器，用于在 overlap scheduling 模式下安全释放内存：

```python
# 文件: python/minisgl/scheduler/cache.py
@contextmanager
def lazy_free_region(self):
    # 延迟释放：在 GPU 执行当前 batch 时
    # 暂不归还页到 free_slots
    # 直到 batch 完成后再统一回收
```

为什么需要延迟释放？在 overlap scheduling 中，CPU 在准备下一个 batch 的同时，GPU 还在执行上一个 batch。如果立即释放上一个 batch 的页，这些页可能被分配给新 batch，导致 GPU 读到被覆盖的数据。`lazy_free_region` 通过推迟释放时机，解决了这个竞态问题。

---

## 10.6 当 GPU 显存耗尽

当所有页都被占用且没有可驱逐的前缀缓存时，系统会进入以下状态：

1. **新 prefill 请求被延迟**：留在 pending 队列，等待 decode 请求完成并释放页
2. **Decode 继续运行**：已在 decode 阶段的请求正常生成 token，每完成一个请求就释放其占用的页
3. **自然恢复**：随着 decode 请求陆续完成，free_slots 逐步回升，pending 的 prefill 请求依次被接纳

Mini-SGLang 没有实现请求抢占（preemption）——即不会中断正在 decode 的请求来为新请求腾出空间。这是对生产系统的一个重要简化。SGLang 完整版支持请求抢占和重计算（recomputation），在极端内存压力下提供更灵活的调度。

---

## 10.7 简化了什么，为什么这样简化

| 特性 | Mini-SGLang | 生产系统 (SGLang/vLLM) |
|-----|-------------|----------------------|
| 页分配 | 简单的 free_slots 集合 | 多层级 Block Allocator |
| 驱逐策略 | LRU 单一策略 | LRU + 优先级 + 引用计数组合 |
| 请求抢占 | 不支持 | 支持 swap/recompute |
| 内存碎片处理 | 无碎片整理 | 定期 compaction |
| 多 GPU 内存 | 不涉及 | 跨 GPU 统一管理 |

Mini-SGLang 保留了内存感知调度的核心骨架——"检查 → 驱逐 → 分配 → 延迟"的决策链路，去掉了生产环境中应对极端场景的复杂机制。对于理解"推理系统如何与 GPU 显存博弈"，这个简化恰到好处。

---

## 本章小结

1. **内存预算**在 Engine 初始化时确定，`page_size` 决定了分配粒度与碎片之间的权衡。
2. **CacheManager** 通过 `allocate_paged` 和 `evict` 实现页的分配与回收，是内存感知的核心。
3. **PrefillAdder** 在准入时逐个检查内存可用性，不足时延迟而非拒绝请求。
4. **分块机制**（ChunkedReq）从 token 维度控制单次 batch 的资源占用。
5. **Lazy Free** 解决了 overlap scheduling 中 CPU/GPU 并行带来的内存竞态问题。
6. **显存耗尽时**系统依赖 decode 完成自然释放，不支持请求抢占——这是对生产系统的有意简化。
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTY5MTg4NTBdfQ==
-->