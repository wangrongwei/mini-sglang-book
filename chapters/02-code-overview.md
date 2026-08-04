# 第 2 章 代码全览

> "在读任何一行实现之前，先画出模块之间的箭头。"

Mini-SGLang 的约 5000 行 Python 代码分布在 10 个顶层模块中。本章不深入任何单个文件的实现细节，而是建立一张全局地图——每个模块做什么、模块之间如何连接、数据从哪里流入又从哪里流出。

---

## 2.1 目录结构

```
python/minisgl/
├── __main__.py              # 启动入口
├── core.py                  # 核心数据结构：SamplingParams, Req, Batch, Context
├── env.py                   # 环境变量配置
├── shell.py                 # 交互式 Shell
│
├── server/                  # HTTP 服务层
│   ├── api_server.py        # FastAPI 路由与 FrontendManager
│   ├── args.py              # 服务器启动参数
│   └── launch.py            # 进程启动编排
│
├── message/                 # 进程间消息定义
│   ├── frontend.py          # 前端消息类型
│   ├── backend.py           # 后端消息类型
│   └── tokenizer.py         # Tokenizer 消息类型
│
├── tokenizer/               # Tokenization / Detokenization
│   ├── tokenize.py
│   ├── detokenize.py
│   └── server.py
│
├── scheduler/               # 调度器
│   ├── scheduler.py         # 主调度循环
│   ├── cache.py             # CacheManager（页式分配）
│   ├── prefill.py           # PrefillManager + PrefillAdder
│   ├── decode.py            # DecodeManager
│   ├── table.py             # TableManager（slot 管理）
│   └── config.py            # 调度器配置
│
├── engine/                  # 推理引擎
│   ├── engine.py            # Engine（模型加载、forward、采样）
│   ├── graph.py             # CUDA Graph 捕获与回放
│   ├── sample.py            # Top-k / Top-p 采样
│   └── config.py            # 引擎配置
│
├── kvcache/                 # KV Cache 管理
│   ├── base.py              # 抽象接口：BaseKVCachePool, BasePrefixCache
│   ├── mha_pool.py          # MHAKVCache（实际张量存储）
│   ├── naive_cache.py       # NaivePrefixCache（无前缀复用）
│   └── radix_cache.py       # RadixPrefixCache（Radix Tree 前缀复用）
│
├── attention/               # Attention 后端
│   ├── base.py              # 抽象接口
│   ├── fa.py                # FlashAttention 后端
│   ├── fi.py                # FlashInfer 后端
│   └── trtllm.py            # TensorRT-LLM 后端
│
├── models/                  # 模型定义
│   ├── llama.py             # LlamaForCausalLM
│   ├── qwen2.py             # Qwen2ForCausalLM
│   ├── qwen3.py             # Qwen3ForCausalLM
│   ├── qwen3_moe.py         # Qwen3MoE
│   ├── mistral.py           # MistralForCausalLM
│   ├── config.py            # 模型配置解析
│   ├── weight.py            # 权重加载
│   └── register.py          # 模型注册表
│
├── layers/                  # 模型层实现
│   ├── attention.py         # Attention 层
│   ├── linear.py            # Linear 层（支持 TP 切分）
│   ├── embedding.py         # Embedding 层
│   ├── norm.py              # RMSNorm
│   ├── rotary.py            # RoPE
│   └── moe.py               # MoE 层
│
├── kernel/                  # 自定义 Kernel
│   ├── index.py             # Index kernel
│   ├── store.py             # KV Store kernel
│   └── radix.py             # Radix Tree C++ 扩展
│
├── distributed/             # 分布式通信
│   ├── impl.py              # NCCL 通信实现
│   └── info.py              # TP 拓扑信息
│
├── moe/                     # MoE 后端
│   ├── base.py
│   └── fused.py
│
└── utils/                   # 工具函数
    ├── logger.py
    ├── torch_utils.py
    └── misc.py
```

---

## 2.2 模块依赖图

模块之间的数据流向如下：

```
  server/api_server.py
        │  (HTTP 请求)
        ▼
  message/ (ZMQ 消息)
        │
        ▼
  tokenizer/  ------------------------------┐
        │  (token IDs)                      │ (detokenized text)
        ▼                                   │
  scheduler/scheduler.py                    │
        │                                   │
        ├──→ scheduler/prefill.py           │
        ├──→ scheduler/decode.py            │
        ├──→ scheduler/cache.py             │
        │        └──→ kvcache/              │
        │                                   │
        ▼                                   │
  engine/engine.py                          │
        │                                   │
        ├──→ models/llama.py (forward)      │
        ├──→ attention/ (attn compute)      │
        ├──→ engine/sample.py (sampling)    │
        └──→ engine/graph.py (CUDA graph)   │
                                            │
        │  (next token IDs)                 │
        ▼                                   │
  scheduler/ -----------------------------►-┘ (更新 Req 状态)
```

关键依赖关系：

- `core.py` 被**所有模块**依赖，是整个系统的数据结构中枢
- `scheduler/` 是 CPU 侧的调度中心，协调 `kvcache/` 的分配与 `engine/` 的计算
- `engine/` 是 GPU 侧的执行中心，调用 `models/` 做 forward，调用 `attention/` 做注意力计算
- `server/` 和 `tokenizer/` 通过 ZMQ 消息队列与 `scheduler/` 进程间通信

---

## 2.3 核心数据结构速览

### 2.3.1 SamplingParams、Req、Batch和Context
整个系统围绕 `core.py` 中定义的四个数据结构运转：

```python
# python/minisgl/core.py

@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0
```

`SamplingParams` 封装用户的采样偏好。注意 `temperature=0.0` 是默认值——即**默认使用贪心解码**。

```python
# 文件: python/minisgl/core.py

@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor   # CPU tensor
    table_idx: int             # 在 page_table 中的行索引
    cached_len: int            # 已缓存的 token 数
    device_len: int            # 已在 GPU 上的 token 数（自动计算）
    output_len: int            # 最大输出 token 数
    uid: int                   # 用户请求唯一标识
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
```

`Req` 是单个请求在调度器中的核心表示。第 3 章将详细分析其生命周期。

```python
# 文件: python/minisgl/core.py

@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode"]
    input_ids: torch.Tensor    # GPU tensor, 由 scheduler 填充
    positions: torch.Tensor    # GPU tensor, 由 scheduler 填充
    out_loc: torch.Tensor      # GPU tensor, 由 scheduler 填充
    padded_reqs: List[Req]     # 填充后的请求列表
    attn_metadata: BaseAttnMetadata  # 由 attention backend 填充
```

`Batch` 是一次 forward pass 的输入打包。`phase` 字段区分 prefill 和 decode 两种计算模式。

```python
# python/minisgl/core.py

@dataclass
class Context:
    page_size: int
    page_table: torch.Tensor
    attn_backend: BaseAttnBackend
    moe_backend: BaseMoeBackend
    kv_cache: BaseKVCachePool
    _batch: Batch | None

    @contextmanager
    def forward_batch(self, batch: Batch):
        ...
```

`Context` 是全局单例，持有 KV Cache、page table、attention 后端等运行时状态。通过 `forward_batch()` 上下文管理器在 forward 期间绑定当前 Batch。

### 2.3.2 PendingReq、TableManager
PendingReq：请求在 CPU 中的存储方式。推理过程，需要拷贝到token_pool 中。
```python
@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens
```

```python
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
```
token_pool：全局词汇池，表示在 GPU 中的请求存储方式。<br>
page_table：token_pool 中对应 token 的 kv 数据存储的显存地址，主要用于底层的 Attention 计算。新请求如果命中了其他 kv cache，会将命中的 page_table 拷贝到新请求的 page_table 中。

```python
class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event

# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"
```

### 2.3.3 消息结构体

```python
@dataclass
class BaseTokenizerMsg:
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    uid: int
```
BaseTokenizerMsg 是一个基类 dataclass，用于定义前端 API Server 与 Tokenizer/后端之间传输的通信消息基础格式。
它本身提供统一的序列化（encoder）与反序列化（decoder）方法（序列化时会自动带上消息类型的名称字段 type），真正的消息格式由继承它的 4 种具体子类 组成：

1. TokenizeMsg（分词/编码请求）：前端收到用户的 HTTP 请求后，打包发送给 Tokenizer 进程的文本编码消息：
- uid (int)：当前请求的唯一 ID。
- text (str | List[Dict[str, str]])：输入的提示词文本，或是 Chat 格式的对话列表（如 [{"role": "user", "content": "..."}]）。
- sampling_params (SamplingParams)：采样参数对象（如 temperature, top_p, max_new_tokens, ignore_eos 等）。
网络传输序列化 JSON/MsgPack 格式示例
以 TokenizeMsg 为例，经过 serialize_type 编码后在 ZMQ 管道传输的 JSON/Dict 结构如下：
```
{
  "type": "TokenizeMsg",
  "uid": 1,
  "text": "Hello, explain SGLang.",
  "sampling_params": {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_new_tokens": 128,
    "ignore_eos": false
  }
}
```

2. DetokenizeMsg（解码结果通知）：后端/调度器生成新 Token 后，回传给前端的解词消息：
- uid (int)：对应请求的唯一 ID。
- next_token (int)：本次推理刚刚生成的单步 Token ID。
- finished (bool)：该请求是否已生成完毕（如遇到 EOS 或达到最大输出长度）。

3. AbortMsg（取消请求）：客户端中途断开连接时，前端发送给后端的取消指令：
- uid (int)：要中断/取消的请求 UID。

4. BatchTokenizerMsg（批量消息包装）：用于将多条消息打包批量发送的容器：
- data (List[BaseTokenizerMsg])：内含的消息列表（例如批量传输多条 TokenizeMsg）。

---

## 2.4 代码密度分析

Mini-SGLang 的代码分布呈现出清晰的层次：

| 模块 | 核心文件数 | 职责 |
|-----|----------|------|
| `core.py` | 1 | 全局数据结构定义 |
| `scheduler/` | 6 | 请求调度、Cache 分配、Batch 构建 |
| `engine/` | 4 | 模型 forward、采样、CUDA Graph |
| `kvcache/` | 4 | KV Cache 存储与前缀缓存 |
| `attention/` | 4 | Attention 计算后端抽象 |
| `models/` | 8 | 模型架构定义与权重加载 |
| `server/` | 3 | HTTP API 与进程启动 |
| `layers/` | 7 | 可复用的模型层组件 |

其中，`scheduler/` 和 `engine/` 是理解推理引擎最关键的两个模块。调度器负责"决定下一步算什么"，引擎负责"实际去算"。

---

## 2.5 入口点

Mini-SGLang 的启动流程始于 `__main__.py`，经由 `server/launch.py` 编排多进程：

```
python -m minisgl --model <path>
    │
    ▼
server/launch.py
    ├── 启动 Tokenizer 进程
    ├── 启动 Scheduler 进程（每个 TP rank 一个）
    └── 启动 Frontend API 进程（FastAPI + uvicorn）
```

每个进程通过 ZMQ IPC 通信。`server/api_server.py` 中的 `FrontendManager` 管理用户连接，将请求序列化为 `TokenizeMsg` 发送给 Tokenizer 进程，Tokenizer 将 token IDs 转发给 Scheduler，Scheduler 调用 Engine 执行推理，结果反向流回。

```python
# 文件: python/minisgl/server/api_server.py

@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )
    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )
```

这是最简单的请求入口：接收 prompt 文本和生成参数，分配 uid，发送消息，返回 SSE 流。

---

## 2.6 与 SGLang 主仓库的结构对比

| 维度 | SGLang | Mini-SGLang |
|------|--------|-------------|
| Router / Gateway | 独立的 Router 进程，支持 DP | 直接在 Frontend 中处理 |
| Tokenizer | 独立进程池 | 单进程 |
| Scheduler | 复杂的多级队列 | 单级 pending list |
| KV Cache | BlockManager + RadixAttention | CacheManager + 可选 RadixPrefixCache |
| Attention | 多后端 + 自动选择 | 保留 FA / FI / TRT-LLM 三后端 |
| 模型 | 60+ 架构 | 5 个架构（Llama/Qwen2/Qwen3/Qwen3MoE/Mistral） |

结构上的对应关系是清晰的：Mini-SGLang 的每个模块都能在 SGLang 主仓库中找到对应物，只是规模和复杂度不同。

---

## 本章小结

1. Mini-SGLang 的 `python/minisgl/` 目录包含 10 个顶层模块，`core.py` 是被所有模块依赖的数据结构中枢
2. 系统分为三个进程：Frontend（HTTP API）、Tokenizer（分词/解词）、Scheduler+Engine（调度+计算），通过 ZMQ 通信
3. 四个核心数据结构——`SamplingParams`、`Req`、`Batch`、`Context`——定义在 `core.py` 中，贯穿整个请求生命周期
4. `scheduler/` 和 `engine/` 是最关键的两个模块：前者决定"算什么"，后者负责"去算"
5. Mini-SGLang 的模块结构与 SGLang 主仓库一一对应，是理解生产代码的最佳跳板
