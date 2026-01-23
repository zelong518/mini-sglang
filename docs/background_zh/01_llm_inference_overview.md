# 01｜LLM 推理系统全景：从请求到 token（面向小白）

> 目标：读完你能回答三件事：
>
> - **用户发一句话**，服务端内部发生了哪些步骤？
> - 为什么 LLM 推理会分成 **prefill** 和 **decode** 两种阶段？
> - 一个“高性能推理系统”到底在优化什么（瓶颈在哪里）？

---

## 1. 一条请求的最简生命周期

以“在线 chat/completions”接口为例，典型流水线如下（概念顺序）：

1. **HTTP API 接收请求**
   - 用户发来 prompt（或 messages）+ 采样参数（max_tokens/temperature…）

2. **Tokenizer：文本 → token ids**
   - LLM 不直接处理文字，而是处理整数序列（token id）

3. **调度器（Scheduler）：决定“这一步 GPU 要算谁”**
   - 在线系统会同时服务很多请求
   - GPU 一次 forward 往往会把多个请求拼成一个 batch

4. **执行引擎（Engine）：在 GPU 上跑 forward**
   - 计算 logits（下一 token 的概率分布）
   - 同时写入/更新 KV cache（后面解释）

5. **Sampler：从 logits 里选出 next token**
   - greedy / temperature / top-k / top-p 等

6. **Detokenizer：token id → 增量文本**
   - 把每步生成的 token “翻译回文字”
   - 以 streaming 方式一点点返回给用户

7. **结束条件**
   - 生成到 `max_tokens`，或遇到 EOS，或被用户取消

你可以把整个系统想成：**消息队列 + 调度器 + GPU 计算内核 + 文本编码/解码** 的组合。

---

## 2. 为什么要分 prefill 和 decode？

这点是所有 Serving 系统的核心。

### 2.1 prefill（提示词预填充）

prefill 要处理的是：用户输入的 prompt（可能很长）。它的特点：

- 一次 forward 可能要“吃掉”很多 token（变长）
- 注意力（attention）在长序列上非常重
- 会产生一大段 KV cache（显存占用大）

### 2.2 decode（自回归生成）

decode 是生成阶段：每次只生成 1 个 token，然后把它 append 到序列末尾，继续下一步。

特点：

- 每步只新增 1 个 token 的计算（相对轻）
- 但要重复很多步（比如生成 512 tokens 就要跑 512 次）
- 系统开销（CPU 调度、kernel launch、内存搬运）会非常显著

因此：prefill 更像“重型任务”，decode 更像“超高频的小任务”。  
这也是为什么系统会有：

- chunked prefill（把重型任务切块）
- CUDA graph（把小任务的 launch 开销压下去）
- overlap scheduling（把 CPU 工作藏到 GPU 时间里）

---

## 3. 推理系统到底在优化什么？

通常你会看到两个指标：

- **吞吐（throughput）**：每秒生成多少 tokens（tokens/s）
- **延迟（latency）**：用户多久看到第一个 token（TTFT）/ 完成响应（E2E）

系统优化几乎都围绕以下瓶颈：

### 3.1 GPU 计算瓶颈

- attention 的算子效率（FlashAttention/FlashInfer）
- GEMM（线性层）吞吐
- 混合精度（fp16/bf16）与 tensor core 使用

### 3.2 GPU 内存瓶颈（KV cache）

KV cache 会随着序列长度线性增长，是显存大头之一。系统需要：

- 尽量复用（prefix cache / radix cache）
- 设计更友好的存储布局（paged KV）
- 控制峰值（chunked prefill）

### 3.3 CPU/调度瓶颈（decode 常见）

decode 每 token 一次 forward，如果每次都要执行大量 Python 逻辑/构造元数据/启动很多 kernel，会导致：

- GPU 等 CPU（utilization 下降）

典型解决方案：

- CUDA graph replay（把 kernel launch 序列固化）
- overlap scheduling（双 stream + pipeline）
- 轻量化消息传递与批处理

---

## 4. Mini-SGLang 把这些概念落在哪？

你不需要马上看懂源码，但先知道“谁负责什么”：

- API：`python/minisgl/server/api_server.py`
- Tokenize/Detokenize：`python/minisgl/tokenizer/server.py`
- 调度：`python/minisgl/scheduler/scheduler.py`
- 执行：`python/minisgl/engine/engine.py`
- KV cache：`python/minisgl/kvcache/*`
- Attention backend：`python/minisgl/attention/*`
- CUDA graph：`python/minisgl/engine/graph.py`

下一篇建议读：

- `02_transformer_attention_basics.md`


