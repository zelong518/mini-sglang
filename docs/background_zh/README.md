# 背景知识（面向推理小白）

这个文件夹用于补齐阅读 `docs/overview_zh.md` 所需的基础知识。每篇尽量“从零开始”，并且与 Mini-SGLang 的代码点对点对应。

## 建议阅读顺序（从 0 到能看懂系统）

1. `01_llm_inference_overview.md`：LLM 推理系统全景（请求到 token）
2. `02_transformer_attention_basics.md`：Transformer/Attention 最小必要知识
3. `03_prefill_vs_decode.md`：prefill vs decode 为什么是两种“完全不同”的负载
4. `04_kv_cache_basics.md`：KV cache 是什么、为什么能加速、为什么会吃显存
5. `05_paged_kv_and_page_table.md`：paged KV/page table 的核心思路（对应 Mini-SGLang）
6. `06_batching_and_scheduling.md`：batching、吞吐/延迟权衡、调度器在做什么
7. `07_sampling_basics.md`：采样（temperature/top-k/top-p）与常见坑
8. `08_cuda_basics_streams.md`：CUDA 基础、stream、异步拷贝、事件
9. `09_cuda_graph_basics.md`：CUDA graph 为何能提升 decode 吞吐
10. `10_tensor_parallel_nccl.md`：Tensor Parallel、NCCL、all-reduce/all-gather
11. `11_zmq_msgpack_ipc.md`：ZMQ + msgpack 的消息队列与进程解耦
12. `12_kernels_aot_jit.md`：AOT vs JIT 自定义内核（tvm-ffi）在系统中的角色

## 与 overview 的关系

`docs/overview_zh.md` 的每个大章节会在合适位置标注：

- “**小白补课**：`docs/background_zh/xxx.md`”

你可以按自己的背景跳读：看不懂再回到这里补齐概念。


