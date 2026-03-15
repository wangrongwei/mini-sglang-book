# 附录 B：关键函数速查

> "Good code is its own best documentation." —— Steve McConnell

本附录按功能分类列出 Mini-SGLang 中所有关键函数和方法，提供文件路径、签名摘要和一句话说明，方便读者在阅读源码时快速定位。

---

## B.1 Core 核心数据结构

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `SamplingParams.__init__()` | `core.py` | `(temperature, top_k, top_p, max_new_tokens)` | 采样参数容器，控制生成行为 |
| `SamplingParams.is_greedy` | `core.py` | `@property -> bool` | 判断是否为贪心采样模式 |
| `Req.__init__()` | `core.py` | `(input_ids, sampling_params, ...)` | 单个推理请求，追踪 Token 位置与缓存状态 |
| `Req.complete_one()` | `core.py` | `() -> None` | 完成一步 Decode，推进输出位置 |
| `Batch.__init__()` | `core.py` | `(reqs, phase, input_ids, positions, out_loc)` | 请求批次，持有 GPU 张量 |
| `Context.__init__()` | `core.py` | `(attn_backend, moe_backend, kv_pool, page_table)` | 全局上下文，持有后端和缓存引用 |
| `Context.forward_batch()` | `core.py` | `(batch) -> contextmanager` | 上下文管理器，设置当前活跃 Batch |
| `set_global_ctx()` | `core.py` | `(ctx: Context) -> None` | 设置全局 Context 单例 |
| `get_global_ctx()` | `core.py` | `() -> Context` | 获取全局 Context 单例 |

---

## B.2 Engine 推理引擎

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `Engine.__init__()` | `engine/engine.py` | `(config, rank, ...)` | 初始化模型、KV Cache、Attention 后端、CUDA Graph |
| `Engine.forward_batch()` | `engine/engine.py` | `(batch: Batch) -> ForwardOutput` | 执行前向推理（CUDA Graph 或直接推理）并异步采样 |
| `Engine._sync_get_memory()` | `engine/engine.py` | `() -> int` | 跨 TP rank 同步可用显存 |
| `Engine._determine_num_pages()` | `engine/engine.py` | `() -> int` | 根据可用显存计算 KV Cache 页数 |
| `_adjust_config()` | `engine/engine.py` | `(config) -> config` | 根据 GPU 架构自动选择 Attention 后端 |
| `GraphRunner.__init__()` | `engine/graph.py` | `(stream, device, model, attn_backend, ...)` | 初始化 CUDA Graph 捕获器 |
| `GraphRunner._capture_graphs()` | `engine/graph.py` | `(max_seq_len, vocab_size, model)` | 为各 Batch Size 捕获 CUDA Graph |
| `GraphRunner.can_use_cuda_graph()` | `engine/graph.py` | `(batch: Batch) -> bool` | 判断当前 Batch 是否可用 CUDA Graph |
| `GraphRunner.replay()` | `engine/graph.py` | `(batch: Batch) -> Tensor` | 回放预捕获的 CUDA Graph |
| `GraphRunner.pad_batch()` | `engine/graph.py` | `(batch: Batch) -> None` | 将 Batch 填充到最近的 CUDA Graph Batch Size |
| `GraphRunner.destroy_cuda_graphs()` | `engine/graph.py` | `() -> None` | 释放所有捕获的 CUDA Graph |
| `GraphCaptureBuffer.init()` | `engine/graph.py` | `(cls, bs, vocab_size, device) -> Self` | 创建 CUDA Graph 捕获时使用的固定张量缓冲区 |
| `GraphCaptureBuffer.copy_from()` | `engine/graph.py` | `(batch: Batch) -> None` | 将 Batch 数据拷贝到捕获缓冲区 |

---

## B.3 Sampling 采样

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `Sampler.prepare()` | `engine/sample.py` | `(batch: Batch) -> None` | 从 Batch 中提取采样参数，构建 GPU 张量 |
| `Sampler.sample()` | `engine/sample.py` | `(logits: Tensor) -> Tensor` | 执行采样：Greedy 或概率采样 |
| `sample_impl()` | `engine/sample.py` | `(logits, args: BatchSamplingArgs) -> Tensor` | 核心采样实现：温度缩放 + Top-K/Top-P 过滤 |
| `make_device_tensor()` | `engine/sample.py` | `(data: list) -> Tensor` | 将 Python 列表转换为 GPU 张量（非阻塞传输） |

---

## B.4 Scheduler 调度器

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `Scheduler.__init__()` | `scheduler/scheduler.py` | `(config, rank, ...)` | 初始化引擎、Cache/Decode/Prefill 管理器 |
| `Scheduler.run_forever()` | `scheduler/scheduler.py` | `() -> None` | 调度器主循环入口 |
| `Scheduler.overlap_loop()` | `scheduler/scheduler.py` | `() -> None` | Overlap 调度模式：流水线化执行与结果处理 |
| `Scheduler.normal_loop()` | `scheduler/scheduler.py` | `() -> None` | 普通调度模式：顺序执行 |
| `Scheduler._schedule_next_batch()` | `scheduler/scheduler.py` | `() -> Batch \| None` | 从 Prefill/Decode 队列选择下一个 Batch |
| `Scheduler._prepare_batch()` | `scheduler/scheduler.py` | `(batch: Batch) -> None` | 为 Batch 构建 GPU 张量（positions、input_ids 等） |
| `Scheduler._process_last_data()` | `scheduler/scheduler.py` | `(batch, output) -> None` | 处理上一轮推理结果：Detokenization 与请求完成 |
| `Scheduler._forward()` | `scheduler/scheduler.py` | `(batch: Batch) -> ForwardOutput` | 调用 Engine 执行前向推理 |
| `_make_positions()` | `scheduler/scheduler.py` | `(batch) -> Tensor` | 构建位置索引张量 |
| `_make_input_tuple()` | `scheduler/scheduler.py` | `(batch) -> Tensor` | 构建 Token 到请求的映射 |
| `_make_write_tuple()` | `scheduler/scheduler.py` | `(batch) -> Tensor` | 构建输出写入位置映射 |

---

## B.5 Prefill 管理

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `PrefillManager.add_one_req()` | `scheduler/prefill.py` | `(req: UserMsg) -> None` | 将用户请求加入待处理队列 |
| `PrefillManager.schedule_next_batch()` | `scheduler/prefill.py` | `(prefill_budget: int) -> Batch \| None` | 从待处理队列组装 Prefill Batch |
| `PrefillManager.abort_req()` | `scheduler/prefill.py` | `(uid: int) -> Req \| None` | 按 UID 中止待处理请求 |
| `PrefillManager.runnable` | `scheduler/prefill.py` | `@property -> bool` | 是否有待处理请求 |
| `PrefillAdder.try_add_one()` | `scheduler/prefill.py` | `(pending_req) -> Req \| None` | 尝试将一个请求加入当前 Batch（检查资源） |
| `PrefillAdder._try_allocate_one()` | `scheduler/prefill.py` | `(req) -> tuple \| None` | 为请求分配缓存资源 |
| `PrefillAdder._add_one_req()` | `scheduler/prefill.py` | `(pending_req, cache_handle, ...) -> Req` | 创建并配置 Req 对象 |
| `ChunkedReq.append_host()` | `scheduler/prefill.py` | `(next_token: Tensor) -> None` | 分块 Prefill 时追加 Token（抛出 NotImplementedError） |

---

## B.6 Decode 管理

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `DecodeManager.schedule_next_batch()` | `scheduler/decode.py` | `() -> Batch \| None` | 从运行中的请求组装 Decode Batch |
| `DecodeManager.filter_reqs()` | `scheduler/decode.py` | `(reqs: Iterable[Req]) -> None` | 过滤并加入可 Decode 的请求 |
| `DecodeManager.remove_req()` | `scheduler/decode.py` | `(req: Req) -> None` | 移除已完成的请求 |
| `DecodeManager.abort_req()` | `scheduler/decode.py` | `(uid: int) -> Req \| None` | 按 UID 中止运行中的请求 |
| `DecodeManager.inflight_tokens` | `scheduler/decode.py` | `@property -> int` | 计算所有运行请求的预留 Token 总数 |
| `DecodeManager.runnable` | `scheduler/decode.py` | `@property -> bool` | 是否有运行中的请求 |

---

## B.7 Cache 管理

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `CacheManager.__init__()` | `scheduler/cache.py` | `(num_pages, page_size, page_table, cache_type)` | 初始化页式缓存管理器 |
| `CacheManager.allocate_paged()` | `scheduler/cache.py` | `(reqs) -> None` | 为多个请求分配页面 |
| `CacheManager.match_req()` | `scheduler/cache.py` | `(req) -> MatchResult` | 匹配请求的前缀缓存 |
| `CacheManager.cache_req()` | `scheduler/cache.py` | `(req, finished) -> None` | 管理请求的缓存生命周期 |
| `CacheManager._allocate()` | `scheduler/cache.py` | `(size) -> list` | 分配空闲页面，必要时触发驱逐 |
| `CacheManager._free()` | `scheduler/cache.py` | `(pages) -> None` | 释放页面到空闲池 |
| `_write_page_table()` | `scheduler/cache.py` | `(page_table, pages, ...) -> None` | 将页面映射写入 Page Table |

---

## B.8 KV Cache Pool

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `BaseKVCachePool.k_cache()` | `kvcache/base.py` | `(index: int) -> Tensor` | 获取第 index 层的 Key 缓存张量 |
| `BaseKVCachePool.v_cache()` | `kvcache/base.py` | `(index: int) -> Tensor` | 获取第 index 层的 Value 缓存张量 |
| `BaseKVCachePool.store_kv()` | `kvcache/base.py` | `(k, v, out_loc, layer_id) -> None` | 将 KV 写入指定位置 |
| `BasePrefixCache.match_prefix()` | `kvcache/base.py` | `(input_ids: Tensor) -> MatchResult` | 在前缀树中匹配已缓存的前缀 |
| `BasePrefixCache.insert_prefix()` | `kvcache/base.py` | `(input_ids, indices) -> InsertResult` | 向前缀树插入新条目 |
| `BasePrefixCache.evict()` | `kvcache/base.py` | `(size: int) -> Tensor` | 驱逐指定数量的缓存页面 |
| `BasePrefixCache.lock_handle()` | `kvcache/base.py` | `(handle, unlock) -> None` | 锁定/解锁缓存句柄防止被驱逐 |

---

## B.9 Attention 后端

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `BaseAttnBackend.forward()` | `attention/base.py` | `(q, k, v, layer_id, batch) -> Tensor` | 执行注意力计算 |
| `BaseAttnBackend.prepare_metadata()` | `attention/base.py` | `(batch) -> None` | 准备 Attention 元数据 |
| `BaseAttnBackend.init_capture_graph()` | `attention/base.py` | `(max_seq_len, bs_list) -> None` | 初始化 CUDA Graph 捕获所需状态 |
| `BaseAttnBackend.prepare_for_capture()` | `attention/base.py` | `(batch) -> None` | 为 CUDA Graph 捕获准备元数据 |
| `BaseAttnBackend.prepare_for_replay()` | `attention/base.py` | `(batch) -> None` | 为 CUDA Graph 回放准备元数据 |
| `BaseAttnMetadata.get_last_indices()` | `attention/base.py` | `(bs: int) -> Tensor` | 获取每个序列最后一个位置的索引 |
| `HybridBackend.forward()` | `attention/base.py` | `(q, k, v, layer_id, batch) -> Tensor` | 按阶段委托给 Prefill 或 Decode 后端 |

---

## B.10 模型配置

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `ModelConfig.from_hf()` | `models/config.py` | `(cls, config: PretrainedConfig) -> ModelConfig` | 从 HuggingFace 配置转换为 Mini-SGLang 配置 |
| `ModelConfig.is_moe` | `models/config.py` | `@property -> bool` | 判断是否为 MoE 模型 |
| `RotaryConfig.__init__()` | `models/config.py` | `(head_dim, rotary_dim, max_position, base, scaling)` | RoPE 旋转位置编码配置 |

---

## B.11 Server 服务层

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `launch_server()` | `server/launch.py` | `(run_shell=False) -> None` | 解析参数并启动多进程服务 |
| `run_api_server()` | `server/api_server.py` | `(args, ...) -> None` | 初始化 FastAPI 并运行 Uvicorn |
| `generate()` | `server/api_server.py` | `(request: GenerateRequest) -> StreamingResponse` | `/generate` 端点处理函数 |
| `stream_chat_completions()` | `server/api_server.py` | `(request: OpenAICompletionRequest) -> StreamingResponse` | `/v1/chat/completions` 流式处理 |
| `parse_args()` | `server/args.py` | `(args: list, run_shell=False) -> (ServerArgs, bool)` | 解析命令行参数为 ServerArgs |
| `FrontendManager` | `server/api_server.py` | dataclass | 管理前端与 Tokenizer 进程的 ZMQ 通信 |

---

## B.12 环境配置

| 函数/方法 | 文件 | 签名摘要 | 说明 |
|---|---|---|---|
| `EnvVar.__init__()` | `env.py` | `(default_value: T, fn: Callable)` | 带类型转换的环境变量 |
| `EnvVar._init()` | `env.py` | `(name: str) -> None` | 从环境读取并解析变量值 |
| `EnvClassSingleton.__init__()` | `env.py` | `() -> None` | 初始化所有环境变量（单例模式） |
| `_TO_BOOL()` | `env.py` | `(x: str) -> bool` | 将字符串转为布尔值 |
| `_PARSE_MEM_BYTES()` | `env.py` | `(mem: str) -> int` | 解析带单位的内存大小（如 "1G"） |

---

## 本附录小结

1. Mini-SGLang 的关键函数分布在 12 个功能模块中，总计约 80 个核心方法。
2. 数据流主线为：`API Server` -> `Scheduler.run_forever()` -> `_schedule_next_batch()` -> `Engine.forward_batch()` -> `Sampler.sample()` -> `_process_last_data()`。
3. 缓存管理形成独立闭环：`CacheManager.match_req()` -> `allocate_paged()` -> `cache_req()` -> `evict()`。
4. 本表可配合附录 A 的对照表使用，在 Mini-SGLang 中定位函数后，快速查找 SGLang/vLLM 中的对应实现。
