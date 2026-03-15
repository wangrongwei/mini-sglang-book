# 第 13 章 运行 Mini-SGLang

> "Talk is cheap. Show me the code." —— Linus Torvalds

经过前面十二章的源码分析，我们已经对 Mini-SGLang 的内部原理有了系统性的理解。本章将回到实践层面，手把手地演示如何安装、启动、调用和测试 Mini-SGLang，让读者能够亲自上手运行整个推理系统。

---

## 13.1 环境准备与安装

### 硬件要求

Mini-SGLang 是一个 GPU 推理引擎，运行时至少需要一块支持 CUDA 的 NVIDIA GPU（Compute Capability >= 8.0，即 A100/H100/RTX 3090 及以上）。建议显存不低于 16 GB，以便加载 7B 参数级别的模型。

### 克隆仓库

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang
```

### 安装依赖

Mini-SGLang 依赖以下核心包：

| 依赖 | 用途 |
|---|---|
| `torch` | 张量计算与 CUDA 运行时 |
| `transformers` | 模型权重加载与 Tokenizer |
| `flash-attn` | FlashAttention 高性能注意力后端 |
| `flashinfer` | Sampling 与 PagedAttention 后端 |
| `fastapi` | HTTP API 服务框架 |
| `uvicorn` | ASGI 异步服务器 |
| `zmq` (pyzmq) | 进程间通信 |

推荐使用 pip 一键安装：

```bash
pip install -e .
```

若需手动安装各依赖，可执行：

```bash
pip install torch transformers flash-attn flashinfer fastapi uvicorn pyzmq
```

> **提示**：`flash-attn` 和 `flashinfer` 的安装可能需要先配置好 CUDA 工具链。如果编译失败，请确认 `nvcc --version` 输出的 CUDA 版本与 PyTorch 版本匹配。

---

## 13.2 启动服务器

Mini-SGLang 的入口点位于 `python/minisgl/__main__.py`，它调用 `launch_server()` 完成多进程启动。

### 基本启动命令

```bash
python -m minisgl --model-path meta-llama/Llama-3.1-8B-Instruct
```

### 常用启动参数

```bash
# 文件: python/minisgl/server/args.py
python -m minisgl \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 1919 \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-running-requests 64 \
    --cuda-graph-max-bs 32 \
    --page-size 16 \
    --memory-ratio 0.85 \
    --attention-backend auto \
    --cache-type radix
```

关键参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model-path` | (必填) | HuggingFace 模型路径或本地目录 |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `1919` | 监听端口 |
| `--dtype` | `auto` | 权重精度：`float16`、`bfloat16`、`float32` |
| `--tensor-parallel-size` | `1` | 张量并行数（需多卡） |
| `--max-running-requests` | 默认值 | 最大并发请求数 |
| `--cuda-graph-max-bs` | 自动 | CUDA Graph 最大 Batch Size |
| `--memory-ratio` | 默认值 | GPU 显存中分配给 KV Cache 的比例 |
| `--dummy-weight` | `false` | 使用随机权重（用于开发测试） |

### 启动流程

`launch_server()` 函数（文件 `python/minisgl/server/launch.py`）使用 Python `multiprocessing` 以 spawn 模式创建多个子进程：

1. **Scheduler 进程**：每个 Tensor Parallel rank 一个，运行推理引擎
2. **Tokenizer 进程**：负责 Tokenization
3. **Detokenizer 进程**：负责反向 Tokenization
4. **API Server**：在主进程中运行 FastAPI + Uvicorn

所有子进程通过 ACK Queue 同步就绪状态，全部就绪后服务器开始接收请求。

---

## 13.3 调用 API

Mini-SGLang 提供三个 HTTP 端点，定义在 `python/minisgl/server/api_server.py` 中。

### /generate 端点

这是 Mini-SGLang 原生的文本生成接口，返回 Server-Sent Events (SSE) 流：

```bash
curl http://localhost:1919/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "max_tokens": 128
  }'
```

### /v1/chat/completions 端点

兼容 OpenAI Chat Completions API 格式，方便与现有工具链集成：

```bash
curl http://localhost:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请用一句话解释什么是 KV Cache。"}
    ],
    "max_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
  }'
```

流式响应会以 `data: {...}\n\n` 的格式逐 Token 返回，客户端可逐行解析。设置 `"stream": false` 可获取完整的一次性响应。

### /v1/models 端点

列出当前加载的模型信息：

```bash
curl http://localhost:1919/v1/models
```

---

## 13.4 Shell 交互模式

除了 API 服务，Mini-SGLang 还提供一个交互式 Shell 模式，非常适合开发调试。Shell 入口位于 `python/minisgl/shell.py`：

```bash
python -m minisgl.shell --model-path meta-llama/Llama-3.1-8B-Instruct
```

也可以在启动时添加 `--shell-mode` 标志：

```bash
python -m minisgl --model-path meta-llama/Llama-3.1-8B-Instruct --shell-mode
```

Shell 模式会自动将 `cuda_graph_max_bs` 和 `max_running_req` 设为 1，并静默日志输出。交互命令：

| 命令 | 说明 |
|---|---|
| 直接输入文本 | 发送消息并获取回复 |
| `/reset` | 清空对话历史 |
| `/exit` | 退出 Shell |

---

## 13.5 环境变量配置

Mini-SGLang 通过 `EnvClassSingleton`（文件 `python/minisgl/env.py`）管理环境变量。所有变量以 `MINISGL_` 为前缀：

```bash
# 文件: python/minisgl/env.py
```

| 环境变量 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `MINISGL_SHELL_MAX_TOKENS` | int | `2048` | Shell 模式最大生成 Token 数 |
| `MINISGL_SHELL_TOP_K` | int | `-1` | Shell 模式 Top-K（-1 为禁用） |
| `MINISGL_SHELL_TOP_P` | float | `1.0` | Shell 模式 Top-P |
| `MINISGL_SHELL_TEMPERATURE` | float | `0.6` | Shell 模式采样温度 |
| `MINISGL_DISABLE_OVERLAP_SCHEDULING` | bool | `False` | 禁用 Overlap 调度 |
| `MINISGL_FLASHINFER_USE_TENSOR_CORES` | bool/None | `None` | 强制 FlashInfer 使用 Tensor Cores |
| `MINISGL_PYNCCL_MAX_BUFFER_SIZE` | mem | `1GB` | PyNCCL 最大缓冲区大小 |

使用示例：

```bash
MINISGL_SHELL_TEMPERATURE=0.3 MINISGL_DISABLE_OVERLAP_SCHEDULING=1 \
  python -m minisgl --model-path meta-llama/Llama-3.1-8B-Instruct --shell-mode
```

---

## 13.6 基础性能测试

### 简单吞吐量测试

可以使用 Python 脚本发送并发请求来测量吞吐量：

```python
import asyncio
import aiohttp
import time

async def send_request(session, prompt, max_tokens=128):
    async with session.post(
        "http://localhost:1919/generate",
        json={"prompt": prompt, "max_tokens": max_tokens}
    ) as resp:
        result = ""
        async for line in resp.content:
            result += line.decode()
        return result

async def benchmark(num_requests=32, max_tokens=128):
    prompts = [f"Tell me about topic {i}" for i in range(num_requests)]
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        tasks = [send_request(session, p, max_tokens) for p in prompts]
        await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    total_tokens = num_requests * max_tokens
    print(f"Requests: {num_requests}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {total_tokens / elapsed:.1f} tokens/s")

asyncio.run(benchmark())
```

### 关键性能指标

在性能测试中，需要关注以下指标：

- **TTFT (Time To First Token)**：从请求发出到首个 Token 返回的延迟，主要受 Prefill 阶段影响
- **TPOT (Time Per Output Token)**：每个输出 Token 的平均延迟，主要受 Decode 阶段影响
- **Throughput**：单位时间内生成的总 Token 数，受 Continuous Batching 和 CUDA Graph 优化影响

### 调优建议

1. **增大 `--max-running-requests`**：提高并发度，提升 Decode 阶段的 GPU 利用率
2. **调整 `--memory-ratio`**：增大 KV Cache 可容纳更多并发请求，但需为模型权重和临时张量预留空间
3. **使用 `--cuda-graph-max-bs`**：启用 CUDA Graph 可显著降低 Kernel Launch 开销
4. **选择合适的 Attention Backend**：不同 GPU 架构适合不同的后端实现

---

## 本章小结

1. Mini-SGLang 通过 `pip install -e .` 完成安装，核心依赖包括 `torch`、`transformers`、`flash-attn`、`flashinfer`、`fastapi` 等。
2. 使用 `python -m minisgl --model-path <path>` 启动服务器，`launch_server()` 会创建 Scheduler、Tokenizer、Detokenizer 等多个子进程。
3. 服务器提供 `/generate`（原生流式）、`/v1/chat/completions`（OpenAI 兼容）、`/v1/models` 三个 API 端点。
4. Shell 模式（`--shell-mode`）提供交互式终端对话，适合开发和快速验证。
5. 环境变量以 `MINISGL_` 前缀管理，通过 `EnvClassSingleton` 单例在启动时自动加载。
6. 性能测试应关注 TTFT、TPOT 和 Throughput 三个核心指标，可通过调整并发数、显存比例和 CUDA Graph 配置进行调优。
