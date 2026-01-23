# Mini-SGLang 功能特性（中文翻译）

> 本文是 `docs/features.md` 的中文翻译版本，并补充少量面向小白的解释与跳转链接。

## 在线服务（Online Serving）

Mini-SGLang 支持在线服务，并提供 **OpenAI 兼容** 的 API Server。它实现了标准的 `/v1/chat/completions` 端点，可以无缝接入现有工具与客户端。

如需查看完整命令行参数与配置选项，请运行：

```bash
python -m minisgl --help
```

## 交互式 Shell 模式（Interactive Shell Mode）

为了演示与测试，Mini-SGLang 提供了交互式 shell 模式。在该模式下，你可以直接在终端输入 prompt，LLM 会实时生成回复。shell 会自动缓存聊天历史以维持上下文；如需清空对话并重新开始，请使用 `/reset`。

示例：

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务（Distributed Serving）

为了在多张 GPU 上扩展性能，Mini-SGLang 支持 **Tensor Parallelism（TP）**。你可以通过 `--tp n` 指定 GPU 数量（并行度）来启用分布式推理。

小白补课：

- `docs/background_zh/10_tensor_parallel_nccl.md`

## 支持的模型（Supported Models）

当前框架支持以下 **dense（稠密）** 模型架构：

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 系列
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 系列

## Chunked Prefill（分块预填充）

Chunked Prefill 是 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 引入的一种技术，默认启用。它会在 prefill 阶段将很长的 prompt 切成更小的 chunk 来处理，从而显著降低峰值显存使用，并减少长上下文服务中的 OOM（显存不足）风险。

chunk 大小可以通过 `--max-prefill-length n` 配置。注意：把 `n` 设得过小（例如 128）通常不推荐，因为可能显著降低性能。

小白补课：

- `docs/background_zh/03_prefill_vs_decode.md`

## Attention 后端（Attention Backends）

Mini-SGLang 集成了高性能 attention kernel，包括 [`FlashAttention`](https://github.com/Dao-AILab/flash-attention) 与 [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)。它支持为 prefill 与 decode 阶段使用不同的后端，从而最大化效率。

例如，在 NVIDIA Hopper GPU 上，默认使用：

- prefill：`FlashAttention3`
- decode：`FlashInfer`

你可以用 `--attn` 指定后端：

- 如果提供两个值（例如 `--attn fa3,fi`），第一个表示 prefill 后端，第二个表示 decode 后端。

小白补课：

- `docs/background_zh/02_transformer_attention_basics.md`
- `docs/background_zh/03_prefill_vs_decode.md`

## CUDA Graph

为了在 decode 阶段尽可能降低 CPU 的 kernel launch 开销，Mini-SGLang 支持捕获并回放 CUDA graphs，默认启用。

你可以通过 `--cuda-graph-max-bs n` 设置 CUDA graph 捕获的最大 batch size。将 `n` 设为 `0` 会禁用该功能。

小白补课：

- `docs/background_zh/08_cuda_basics_streams.md`
- `docs/background_zh/09_cuda_graph_basics.md`

## Radix Cache（前缀复用缓存）

Mini-SGLang 采用了 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，实现了一个 **Radix Cache** 来管理 KV cache。它能够在不同请求共享前缀时复用已有 KV cache，从而减少重复计算。

该功能默认启用；你也可以通过 `--cache naive` 切换为朴素缓存策略（不做前缀复用）。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Radix Attention 示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/)。*

小白补课：

- `docs/background_zh/04_kv_cache_basics.md`

## Overlap Scheduling（重叠调度）

为了进一步降低 CPU 开销，Mini-SGLang 采用了 [NanoFlow](https://arxiv.org/abs/2408.12757) 提出的 overlap scheduling：把 CPU 调度/元数据处理与 GPU 计算重叠，从而提升整体吞吐。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Overlap Scheduling 示意图，来自 [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。*

小白补课：

- `docs/background_zh/08_cuda_basics_streams.md`
- `docs/background_zh/06_batching_and_scheduling.md`


