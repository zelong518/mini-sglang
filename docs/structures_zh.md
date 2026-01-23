# Mini-SGLang 系统结构（中文翻译）

> 本文是 `docs/structures.md` 的中文翻译版本，并补充少量小白说明与跳转链接。

## 系统架构（System Architecture）

Mini-SGLang 被设计为一个分布式系统，用于高效执行大语言模型（LLM）推理。系统由多个相互独立的进程协同工作。

### 关键组件（Key Components）

- **API Server**：用户入口。提供 OpenAI 兼容 API（例如 `/v1/chat/completions`），接收 prompt 并返回生成文本。
- **Tokenizer Worker**：将输入文本转换成模型可理解的数字（token）。
- **Detokenizer Worker**：将模型生成的数字（token）转换回可读文本。
- **Scheduler Worker**：核心 worker 进程。在多 GPU 配置下，每张 GPU 对应一个 Scheduler Worker（称为一个 **TP Rank**）。它负责该 GPU 的计算与资源分配。

小白补课：

- `docs/background_zh/01_llm_inference_overview.md`
- `docs/background_zh/11_zmq_msgpack_ipc.md`
- `docs/background_zh/10_tensor_parallel_nccl.md`

### 数据流（Data Flow）

组件之间使用：

- **ZeroMQ（ZMQ）**：传递控制消息（control messages）
- **NCCL**（通过 `torch.distributed`）：在 GPU 之间交换“重型张量数据”（heavy tensor data exchange）

![Process overview diagram](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期（Request Lifecycle）：**

1. **用户（User）** 向 **API Server** 发起请求。
2. **API Server** 将请求转发给 **Tokenizer**。
3. **Tokenizer** 将文本转换成 tokens，并发送给 **Scheduler（Rank 0）**。
4. **Scheduler（Rank 0）** 将请求广播给所有其它 Scheduler（如果使用多 GPU）。
5. **所有 Scheduler** 对请求进行调度，并触发各自本地的 **Engine** 计算下一 token。
6. **Scheduler（Rank 0）** 收集输出 token，并发送给 **Detokenizer**。
7. **Detokenizer** 将 token 转换为文本，并回传给 **API Server**。
8. **API Server** 将结果以流式方式返回给 **用户**。

> 小白提示：这里的“广播请求”属于 **控制面同步**（让所有 TP rank 的 scheduler 看到一致的请求序列），与 NCCL 的 **数据面通信**（张量 all-reduce/all-gather）是两件事。

## 代码组织（`minisgl` 包）

源码位于 `python/minisgl/`。下面是面向开发者的模块划分：

- `minisgl.core`：提供核心数据结构 `Req`、`Batch`（请求与 batch 的状态），`Context`（推理上下文的全局状态），以及 `SamplingParams`（用户采样参数）。
- `minisgl.distributed`：提供 TP 下 all-reduce/all-gather 等通信接口；`DistributedInfo` 记录 TP worker 的 rank/size 信息。
- `minisgl.layers`：实现用于构建 LLM 的基础层（带 TP 支持），包括 linear、layernorm、embedding、RoPE 等；通用基类位于 `minisgl.layers.base`。
- `minisgl.models`：实现具体模型（包括 Llama、Qwen3），以及从 Hugging Face 加载与切分权重的工具。
- `minisgl.attention`：定义 attention backend 接口，并实现 `flashattention`/`flashinfer` 等后端；由 `AttentionLayer` 调用，并使用 `Context` 中保存的 metadata。
- `minisgl.kvcache`：定义 KV cache pool/manager 接口，并实现 `MHAKVCache`、`NaiveCacheManager`、`RadixCacheManager`。
- `minisgl.utils`：通用工具集合，包括 logger、以及对 ZMQ 的封装。
- `minisgl.engine`：实现 `Engine`（单进程 TP worker）。负责管理模型、context、KV cache、attention backend，以及 CUDA graph replay。
- `minisgl.message`：定义 api_server/tokenizer/detokenizer/scheduler 之间通过 ZMQ 交换的消息类型；消息支持自动序列化/反序列化。
- `minisgl.scheduler`：实现 `Scheduler`（每个 TP worker 进程运行一个）。rank0 scheduler 接收 tokenizer 的消息，并与其它 ranks 通信，将结果发送给 detokenizer。
- `minisgl.server`：定义 CLI 参数与 `launch_server`（启动 Mini-SGLang 的所有子进程）；实现 FastAPI 前端服务器（`minisgl.server.api_server`），提供 `/v1/chat/completions` 等端点。
- `minisgl.tokenizer`：实现 `tokenize_worker`，处理 tokenization/detokenization 请求。
- `minisgl.llm`：提供 `LLM` 类，作为更易用的 Python 接口与系统交互。
- `minisgl.kernel`：自定义 CUDA kernels；通过 `tvm-ffi` 提供 Python binding 与 JIT 接口。
- `minisgl.benchmark`：benchmark 工具。


