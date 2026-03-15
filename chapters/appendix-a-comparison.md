# 附录 A：与 SGLang/vLLM 的代码对照表

> "Standing on the shoulders of giants." —— Isaac Newton

本附录提供 Mini-SGLang 与两大生产级推理引擎——SGLang（完整版）和 vLLM——的代码对照关系。读者在阅读 Mini-SGLang 源码后，可以借助本表快速定位生产系统中的对应实现，进而深入学习更复杂的优化策略。

---

## A.1 核心数据结构

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `core.py:SamplingParams` | `srt/sampling/sampling_params.py:SamplingParams` | `vllm/sampling_params.py:SamplingParams` |
| `core.py:Req` | `srt/managers/schedule_batch.py:Req` | `vllm/sequence.py:Sequence` |
| `core.py:Batch` | `srt/managers/schedule_batch.py:ScheduleBatch` | `vllm/sequence.py:SequenceGroup` |
| `core.py:Context` | `srt/model_executor/forward_batch_info.py:ForwardBatch` | `vllm/worker/model_runner.py:ModelInput` |

---

## A.2 调度器

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `scheduler/scheduler.py:Scheduler` | `srt/managers/scheduler.py:Scheduler` | `vllm/core/scheduler.py:Scheduler` |
| `scheduler/scheduler.py:overlap_loop()` | `srt/managers/scheduler.py:_overlap_*` | N/A (vLLM 使用不同的 Overlap 策略) |
| `scheduler/scheduler.py:normal_loop()` | `srt/managers/scheduler.py:run_batch()` | `vllm/core/scheduler.py:schedule()` |
| `scheduler/scheduler.py:_schedule_next_batch()` | `srt/managers/scheduler.py:get_next_batch_to_run()` | `vllm/core/scheduler.py:_schedule()` |
| `scheduler/scheduler.py:_process_last_data()` | `srt/managers/scheduler.py:process_batch_result()` | `vllm/engine/async_llm_engine.py:_process_model_outputs()` |

---

## A.3 Prefill 与 Decode 管理

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `scheduler/prefill.py:PrefillManager` | `srt/managers/scheduler.py` (内联) | `vllm/core/scheduler.py:SchedulerPrefillOutputs` |
| `scheduler/prefill.py:PrefillAdder` | `srt/managers/schedule_batch.py:SchedulePolicy` | `vllm/core/scheduler.py:_schedule_prefills()` |
| `scheduler/prefill.py:PrefillAdder.try_add_one()` | `srt/managers/scheduler.py:add_one_req()` | `vllm/core/scheduler.py:_schedule_prefills()` |
| `scheduler/decode.py:DecodeManager` | `srt/managers/scheduler.py` (内联) | `vllm/core/scheduler.py:SchedulerRunningOutputs` |
| `scheduler/decode.py:schedule_next_batch()` | `srt/managers/scheduler.py:get_next_batch_to_run()` | `vllm/core/scheduler.py:_schedule_running()` |

---

## A.4 KV Cache 与内存管理

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `kvcache/base.py:BaseKVCachePool` | `srt/mem_cache/memory_pool.py:BaseTokenToKVPool` | `vllm/core/block_manager.py:BlockSpaceManager` |
| `kvcache/base.py:BaseCacheHandle` | `srt/mem_cache/memory_pool.py` (内部结构) | `vllm/block.py:PhysicalTokenBlock` |
| `scheduler/cache.py:CacheManager` | `srt/mem_cache/radix_cache.py:RadixCache` | `vllm/core/block/cpu_gpu_block_allocator.py` |
| `scheduler/cache.py:CacheManager.allocate_paged()` | `radix_cache.py:alloc()` | `block_manager.py:allocate()` |
| `scheduler/cache.py:CacheManager.evict()` | `radix_cache.py:evict()` | `block_manager.py:free()` |
| `scheduler/cache.py:CacheManager.match_prefix()` | `radix_cache.py:match_prefix()` | N/A (vLLM v1 无 Prefix Caching；v2 在 `prefix_caching.py`) |
| `scheduler/cache.py:CacheManager.insert_prefix()` | `radix_cache.py:insert()` | N/A |
| `kvcache/base.py:BasePrefixCache` | `srt/mem_cache/radix_cache.py:RadixCache` | `vllm/core/block/prefix_caching.py` |

---

## A.5 引擎与模型执行

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `engine/engine.py:Engine` | `srt/model_executor/model_runner.py:ModelRunner` | `vllm/worker/model_runner.py:ModelRunner` |
| `engine/engine.py:Engine.__init__()` | `model_runner.py:__init__()` | `model_runner.py:__init__()` |
| `engine/engine.py:forward_batch()` | `model_runner.py:forward()` | `model_runner.py:execute_model()` |
| `engine/graph.py:GraphRunner` | `srt/model_executor/cuda_graph_runner.py` | `vllm/worker/model_runner.py:CUDAGraphRunner` |
| `engine/graph.py:GraphRunner.capture()` | `cuda_graph_runner.py:capture()` | `CUDAGraphRunner.capture()` |
| `engine/graph.py:GraphRunner.replay()` | `cuda_graph_runner.py:replay()` | `CUDAGraphRunner.forward()` |
| `engine/sample.py:Sampler` | `srt/layers/sampler.py:Sampler` | `vllm/model_executor/layers/sampler.py:Sampler` |

---

## A.6 Attention 后端

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `attention/base.py:BaseAttnBackend` | `srt/layers/attention/base_attention.py` | `vllm/attention/backends/abstract.py:AttentionBackend` |
| `attention/base.py:BaseAttnMetadata` | `srt/layers/attention/base_attention.py:AttentionMetadata` | `vllm/attention/backends/abstract.py:AttentionMetadata` |
| `attention/base.py:HybridBackend` | `srt/layers/attention/` (多后端组合) | `vllm/attention/selector.py` |
| `attention/fa.py` (FlashAttention) | `srt/layers/attention/flashattention_backend.py` | `vllm/attention/backends/flash_attn.py` |

---

## A.7 模型定义与配置

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `models/config.py:ModelConfig` | `srt/model_config.py:ModelConfig` | `vllm/config.py:ModelConfig` |
| `models/config.py:ModelConfig.from_hf()` | `model_config.py:__init__()` | `config.py:ModelConfig.__init__()` |
| `models/config.py:RotaryConfig` | `model_config.py` (内部字段) | `config.py` (内部字段) |
| `models/llama.py` | `srt/models/llama.py` | `vllm/model_executor/models/llama.py` |

---

## A.8 服务器与 API

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `server/api_server.py` | `srt/entrypoints/openai/api_server.py` | `vllm/entrypoints/openai/api_server.py` |
| `server/api_server.py:generate()` | `api_server.py:generate()` | `api_server.py:create_completion()` |
| `server/api_server.py:stream_chat_completions()` | `api_server.py:v1_chat_completions()` | `api_server.py:create_chat_completion()` |
| `server/launch.py:launch_server()` | `srt/server.py:launch_server()` | `vllm/entrypoints/openai/api_server.py:run_server()` |
| `server/args.py:ServerArgs` | `srt/server_args.py:ServerArgs` | `vllm/engine/arg_utils.py:EngineArgs` |
| `shell.py` | N/A | N/A |

---

## A.9 环境配置与工具

| Mini-SGLang | SGLang (完整版) | vLLM |
|---|---|---|
| `env.py:EnvClassSingleton` | `srt/utils.py` (散布的环境变量) | `vllm/envs.py` |
| `env.py:EnvVar` | N/A (直接 `os.getenv`) | `vllm/envs.py:environment_variables` |
| `__main__.py` | `srt/entrypoints/openai/api_server.py:__main__` | `vllm/entrypoints/openai/api_server.py:__main__` |

---

## A.10 架构差异说明

虽然三者在概念上高度对应，但实现复杂度差异显著：

| 特性 | Mini-SGLang | SGLang | vLLM |
|---|---|---|---|
| 代码量级 | ~3,000 行 | ~100,000 行 | ~200,000 行 |
| Tensor Parallelism | 基础支持 | 完整 TP/PP/EP | 完整 TP/PP |
| Prefix Caching | RadixAttention 简化版 | RadixAttention 完整版 | Hash-based Block Caching |
| 调度策略 | Overlap / Normal | Overlap + Chunked Prefill | Chunked Prefill + Preemption |
| 量化支持 | 无 | GPTQ/AWQ/FP8 | GPTQ/AWQ/FP8/INT8 |
| Speculative Decoding | 无 | 支持 | 支持 |
| 多模态 | 无 | 支持 | 支持 |
| 分布式推理 | 单机多卡 | 多机多卡 | 多机多卡 |

---

## 本附录小结

1. Mini-SGLang 与 SGLang 完整版的对应关系最为直接，后者可视为前者在各个维度上的扩展与工程化。
2. vLLM 采用了不同的命名体系和架构组织，但核心概念（请求、批次、调度器、缓存管理器、模型执行器）是一致的。
3. Mini-SGLang 将 SGLang 中散布在多个文件的功能（如 Prefill/Decode 管理）独立为专门模块，更易于理解。
4. 生产系统在量化、多模态、Speculative Decoding 等方面有大量 Mini-SGLang 未涵盖的功能，读者可以在理解基础架构后逐步深入。
