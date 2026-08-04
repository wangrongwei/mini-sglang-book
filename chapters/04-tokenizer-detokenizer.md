# 第 4 章 KV Cache 的简化实现

> "推理引擎的大部分复杂度，都藏在 KV Cache 的管理里。"

Autoregressive 生成的每一步都需要对所有历史 token 做 attention 计算。如果每次都重新计算所有 token 的 Key 和 Value，计算量将随序列长度平方增长。KV Cache 的核心思想是：**将已计算的 Key/Value 张量缓存起来，每步只计算新增 token 的 KV，再与缓存拼接做 attention**。这是 LLM 推理引擎中最基础也最关键的优化。

本章剖析 Mini-SGLang 中 KV Cache 的存储层（`kvcache/`）和分配层（`scheduler/cache.py`），理解它如何用简化的页式管理实现生产级的功能。

---

## 4.1 为什么 KV Cache 需要精心管理

一个直觉的 KV Cache 实现是：为每个请求分配一个 `(seq_len, num_heads, head_dim)` 的连续张量。但这在 serving 场景下有严重问题：

| 问题 | 描述 |
|------|------|
| 碎片化 | 请求的序列长度各异且动态增长，连续分配导致 GPU 内存碎片 |
| 预分配浪费 | 按 `max_tokens` 预分配会浪费大量内存（多数请求不会用满） |
| 无法共享 | 相同 prompt 前缀的 KV 无法跨请求复用 |
| 动态 batch | Continuous Batching 要求请求随时加入/退出，内存管理必须支持细粒度分配回收 |

生产级方案（如 vLLM 的 PagedAttention）借鉴操作系统的虚拟内存机制，将 KV Cache 按**页（page/block）**管理。Mini-SGLang 沿用了这一思路，但大幅简化了实现。

---

## 4.2 存储层：BaseKVCachePool 与 MHAKVCache

KV Cache 的物理存储由 `BaseKVCachePool` 抽象接口定义：

```python
# 文件: python/minisgl/kvcache/base.py

class BaseKVCachePool(ABC):
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...
```

接口很小：读取指定层的 K/V 缓存，以及按指定位置写入新的 K/V。`out_loc` 参数是关键——它是一个整数张量，指定新计算的 KV 应该写入缓存的哪些物理 slot。

具体实现 `MHAKVCache`（在 `kvcache/mha_pool.py` 中）预分配一个巨大的统一 buffer：

```python
# 文件: python/minisgl/kvcache/mha_pool.py（概念）
# buffer 形状:
#   (2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
#    ↑
#    K=0, V=1
```

这个六维张量在引擎初始化时一次性分配，占据 GPU 显存的主要部分。`num_pages` 的计算方式是：

```
可用显存 = 总显存 - 模型权重占用 - 预留安全余量
num_pages = 可用显存 / (2 × num_layers × page_size × kv_heads × head_dim × dtype_size)
```

所有请求共享这个全局 buffer，通过 `page_table` 间接寻址各自的 KV 数据。

---

## 4.3 Page Table：虚拟到物理的映射
虚拟是什么？物理又是什么？

`Context` 持有的 `page_table` 是整个 KV Cache 管理的核心数据结构：

```python
# 文件: python/minisgl/core.py

@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor   # shape: (max_running_reqs, max_seq_len)
    ...
```

`page_table[table_idx, position]` 的值是一个整数，表示请求 `table_idx` 的第 `position` 个 token 的 KV 存储在全局 buffer 的第几个 slot。这就是一层间接寻址：

```
请求视角:  token_0  token_1  token_2  token_3
              │        │        │        │
page_table:  [42]     [43]     [100]    [101]
              │        │        │        │
物理 buffer:  slot42   slot43   slot100  slot101
```

注意 `page_table` 的注释写道"this table always treat page_size = 1"——即 `page_table` 是按 **token 粒度**索引的，若 page_size=16，则上面的token_2、token_3必须接着使用[44]、[45]，即底层物理存储必须按页对齐。这是 Mini-SGLang 的一个重要简化：page table 本身不需要理解页的概念，页对齐的逻辑完全交给分配器。

### 4.3.1 page table 的初始化

```
# NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
self.max_seq_len = min(config.max_seq_len, num_tokens)
aligned_max_seq_len = _align_up_32(self.max_seq_len)

self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
    (config.max_running_req + 1, aligned_max_seq_len),
    dtype=torch.int32,
    device=self.device,
)
```
它是一个在 GPU 上预分配的全局 2D Tensor（二维数组），类型为 int32。
1. 行数 (Row)：max_running_req + 1
    - 代表系统能同时处理的最大并发请求数（额外加 1 个是给 Dummy 请求占位用的）。
    - 每个 Request（根据 req.table_idx）独占其中一行。
2. 列数 (Column)：aligned_max_seq_len
    - 代表模型支持的最大上下文长度（比如 4096），并且在底层做了对齐（_align_up_32，保证每行字节数是 128 bytes 的整数倍，这是为了 GPU 访存效率）。
    - 它的每一列对应的是这个句子里的每一个具体的 Token。
3. 内容 (Value)：Raw Locations (Token 的物理槽位号)
    - 这是本项目最大的特点！ 传统的 PagedAttention，表里填的是“物理页号”。但在这里，填的是打平后的“物理 Token 的绝对存储下标”。

### 4.3.2 结合CacheManager源码的具体举例
为了看懂这个过程，必须结合 CacheManager 里的 _page_to_token 函数：
```
    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        ...
```
它负责把“按页分配出来的地”，强行铺平成一个个连续的 Token 坐标。举例推演：
假设：
- config.page_size = 16（每个物理页装 16 个 Token，vLLM 默认是 16）
- 全局请求表只有 2 行，最大长度 32。
- 初始时，所有位置填满 0（或 -1）。

请求 A 到来：
1. 内容长度为 3 个 Token。
2. 虽然只有 3 个 Token，但 CacheManager 最低也是按“页”批地的（div_ceil 向上取整），所以它分配了 1 个物理页（所以page_size 设置太大，一个小请求就占了一个物理页，导致碎片化严重）。
3. 假设 CacheManager 从空闲池 free_slots 拿到了第 1 页。由于代码里 free_slots 存的其实就是乘过 page_size 的偏移量，所以拿到的是数字 16。
4. 关键一步 (_page_to_token)：把偏移量 16 展开成 16 个 Token 的绝对物理坐标，即 [16, 17, 18, 19, ..., 31]。
5. 然后执行 _write_page_table，把这些坐标填入请求 A 所在的第 0 行的前 16 列：
当前 page_table (简写部分列):
```
       Token_0  Token_1  Token_2  Token_3  ... Token_15  Token_16
Row 0: [  16,      17,      18,      19,   ...    31,       0 ... ] 
Row 1: [   0,       0,       0,       0,   ...     0,       0 ... ]
```
虽然只用了 3 个 Token（占用 16, 17, 18 三个槽位），但在 page_table 里会把这一整页 16 个槽位的映射关系全写上去（因为物理页是不可分割的）。
请求 B 到来：
前缀树 prefix_cache 匹配发现，请求 B 完全可以复用请求 A 的前 3 个 Token，然后它还需要自己生成剩下的词。
这时，请求 B 会分配到第 1 行 (Row 1)：
1. 它的前 3 列，会直接抄作业，填入被复用的物理槽位 [16, 17, 18]。
2. 假设给它新批的地是第 2 页（偏移 32），那么从它的第 3 列开始，就会填入新展开的坐标 [32, 33, 34...]。
```
当前 page_table (简写部分列):
       Token_0  Token_1  Token_2  | Token_3  Token_4 ...
Row 0: [  16,      17,      18,   |   19,      20,   ... ]  <-- Req A 
Row 1: [  16,      17,      18,   |   32,      33,   ... ]  <-- Req B (复用了 Req A 的 16~18 槽位)
```
mini-sglang 这么设计的原因是：传统的做法（表里只存 1 个数字：物理页号），GPU 在每次访问某个 Token 时，都要执行：真实的物理地址 = 页表[逻辑页号] * page_size + 逻辑 Token 偏移量。现在，CPU 端（_page_to_token）把这个算术题提前做好，直接把“绝对坐标”铺在了 page_table 上。所以 GPU 上的 FlashAttention 或 PagedAttention Kernel 在运行时，只需要执行极简的寻址：真实的物理地址 = 页表[当前 Token 索引]省去了乘法和加法。


---

## 4.4 分配层：CacheManager

`CacheManager`（在 `scheduler/cache.py` 中）负责 KV Cache slot 的分配与回收：

```python
# 文件: python/minisgl/scheduler/cache.py

class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        self.num_pages = num_pages
        self.page_table = page_table
        self.page_size = page_size
```

**`free_slots` 的设计**：它是一个一维整数张量，存储所有空闲页的起始 token 索引。例如 `page_size=4` 时，`free_slots = [0, 4, 8, 12, ...]`。每个元素代表一个空闲页的首地址。

### 分配流程

当一批请求需要新的 KV Cache 空间时，`allocate_paged()` 被调用：

```python
# python/minisgl/scheduler/cache.py
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
        --> decode_manager/schedule_next_batch
        --> _prepare_batch
            --> allocate_paged

def allocate_paged(self, reqs: List[Req]) -> None:
    needed_pages = 0
    allocation_info: List[Tuple[int, int, int]] = []
    for req in reqs:
        first_page = div_ceil(req.cached_len, self.page_size)
        last_page = div_ceil(req.device_len, self.page_size)
        if last_page > first_page:
            needed_pages += last_page - first_page
            allocation_info.append((req.table_idx, first_page, last_page))
    if needed_pages > 0:
        allocated = self._page_to_token(self._allocate(needed_pages))
        _write_page_table(self.page_table, allocated, allocation_info, self.page_size)
```

流程分三步：

1. **统计需求**：对每个请求，计算从 `cached_len` 到 `device_len` 跨越了多少新页
2. **批量分配**：从 `free_slots` 中取出足够的空闲页（不够时触发 eviction）
3. **写入 page_table**：将分配到的物理 slot 写入对应请求的 `page_table` 行

### 回收与 Eviction

当空闲页不足时，`_allocate()` 会触发前缀缓存的 eviction：

```python
# python/minisgl/scheduler/cache.py
run_forever
  --> overlap_loop
    --> _process_one_msg
    --> _schedule_next_batch
        --> prefill manager/schedule_next_batch
        --> decode_manager/schedule_next_batch
        --> _prepare_batch
            --> allocate_paged
                --> _allocate
                    --> prefix_cache.evict # 如果free_slots不足，先尝试驱逐；

def _allocate(self, needed_pages: int) -> torch.Tensor:
    if needed_pages > (free_pages := len(self.free_slots)):
        evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
        self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
    allocated = self.free_slots[:needed_pages]
    self.free_slots = self.free_slots[needed_pages:]
    return allocated
```

这里体现了 Mini-SGLang 的内存管理策略：**空闲页优先，不够时从前缀缓存中逐出最久未使用的条目**。`evicted[:: self.page_size]` 的步长切片将 token 粒度的索引转换回页粒度。
```python
    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        leave_nodes = self._collect_leave_nodes_for_evict()
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            evicted_size += node.length
            evicted_indices.append(node.value)
            self.evictable_size -= node.length
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        return torch.cat(evicted_indices)
```


---

## 4.5 前缀缓存：BaseCacheHandle 接口

`BaseCacheHandle` 是请求与前缀缓存之间的桥梁：

```python
# 文件: python/minisgl/kvcache/base.py

@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...
```

`cached_len` 记录前缀缓存命中了多少 token，`get_matched_indices()` 返回命中部分在全局 buffer 中的物理 slot 索引。`frozen=True` 表明 handle 一旦创建就不可变——这是因为 Radix Tree 可能随时重组节点，handle 只是一个快照。

Mini-SGLang 提供两种前缀缓存实现：

### NaivePrefixCache：无缓存基线

```python
# 文件: python/minisgl/kvcache/naive_cache.py

class NaiveCacheHandle(BaseCacheHandle):
    empty_tensor: torch.Tensor

    def __init__(self):
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor

class NaivePrefixCache(BasePrefixCache):
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids, indices) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor
        raise NotImplementedError("NaiveCacheManager does not support eviction.")
```

`NaivePrefixCache` 是最简基线：永远报告"没有缓存命中"（`cached_len=0`），不存储任何前缀。所有 KV 空间在请求结束后直接释放回 `free_slots`。这等同于关闭前缀复用功能。

### RadixPrefixCache：Radix Tree 前缀复用

`RadixPrefixCache`（在 `kvcache/radix_cache.py` 中）使用 Radix Tree 数据结构存储已完成请求的 token 序列及其 KV Cache 映射。当新请求的 prompt 与已缓存前缀匹配时，命中部分无需重新计算——对应的 KV 数据已经在全局 buffer 中。

两者的对比：

| 特性 | NaivePrefixCache | RadixPrefixCache |
|------|-----------------|-----------------|
| 前缀匹配 | 永远返回 0 | Radix Tree 最长匹配 |
| KV 复用 | 无 | 跨请求复用相同前缀的 KV |
| Eviction | 不支持 | LRU 策略逐出叶节点 |
| 适用场景 | 调试/基线测试 | 生产运行 |

---

## 4.6 page_size 的作用

`page_size` 决定了分配的粒度。Mini-SGLang 支持 `page_size >= 1`：

- **`page_size=1`**：每个 token 独立分配，最灵活但管理开销最大
- **`page_size>1`**：按页分配，减少管理开销但可能有尾部浪费

`_page_to_token()` 方法展示了页到 token 的展开逻辑：

```python
# 文件: python/minisgl/scheduler/cache.py

def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
    if self.page_size == 1:
        return pages
    offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
    return (pages.unsqueeze(1) + offsets).flatten()
```

例如 `page_size=4`，一个页地址 `[8]` 会被展开为 `[8, 9, 10, 11]`，对应 4 个连续的 token slot。

---

## 4.7 与生产实现的对比

Mini-SGLang 的 KV Cache 管理相比生产级实现做了以下简化：

| 维度 | 生产级 SGLang/vLLM | Mini-SGLang |
|------|-------------------|-------------|
| Block 管理 | 复杂的 BlockManager，支持 CoW、swap to CPU | `free_slots` 张量 + 简单分配回收 |
| Page Table 更新 | 批量异步更新，最小化 CPU-GPU 同步 | `_write_page_table` 同步写入 |
| 内存池 | 多级缓存（GPU L1/L2/HBM/Host） | 单级 GPU HBM |
| Eviction 策略 | 多种策略（LRU/LFU/FIFO） | LRU（Radix Tree 叶节点） |
| 碎片整理 | 定期 compaction | 无 |

核心差异在于 **`free_slots` 张量替代了 BlockManager 的整个管理逻辑**。生产级 BlockManager 需要处理 Copy-on-Write（当前缀被共享时）、跨设备 swap、碎片整理等问题。Mini-SGLang 将这些全部省略，用一个一维张量的头部弹出/尾部追加实现分配和回收。

这种简化之所以可行，是因为 Mini-SGLang 的 Radix Prefix Cache 已经处理了最重要的优化（前缀复用），而 Copy-on-Write、swap 等功能对理解核心机制不是必需的。

---

## 4.8 完整的 KV Cache 写入路径

将上述组件串联起来，一次 prefill 的 KV Cache 写入路径如下：

```
1. PrefillAdder 为请求分配 table_idx（TableManager.allocate）
2. CacheManager.allocate_paged 分配新页 → 写入 page_table
3. Engine.forward_batch 执行模型 forward
   → 每个 Attention 层调用 kv_cache.store_kv(k, v, out_loc, layer_id)
   → out_loc 来自 page_table[table_idx, cached_len:device_len]
4. K/V 数据写入全局 buffer 的对应 slot
5. Prefill 完成后，req.cached_len 更新为 req.device_len
```

Decode 阶段的路径相同，只是每轮只写入 1 个 token 的 KV（`extend_len=1`）。

---

## 本章小结

1. KV Cache 是 autoregressive 生成的核心优化，避免每步重新计算所有历史 token 的 Key/Value
2. `BaseKVCachePool` 定义了存储层接口，`MHAKVCache` 预分配一个六维全局 buffer，所有请求共享
3. `page_table` 实现虚拟到物理的映射——`page_table[table_idx, position]` 记录每个 token 的 KV 物理 slot
4. `CacheManager` 通过 `free_slots` 张量管理空闲页，不足时从前缀缓存 evict，整个分配逻辑不到 30 行代码
5. Mini-SGLang 用 `free_slots` 张量替代了生产级 BlockManager 的复杂管理逻辑，省略了 Copy-on-Write、swap、碎片整理等功能，但保留了页式分配和前缀复用这两个最核心的机制
