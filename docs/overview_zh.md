# Mini-SGLang 项目中文总览（超详细）

> 本文面向想“看懂/改懂/复现”现代 LLM Serving 系统的读者：它不仅解释 **Mini-SGLang 是什么**，更会把 **从请求到 token** 的完整链路、核心数据结构与性能优化的“原理 + 代码落点”讲清楚。
>
> - 代码根目录：`python/minisgl/`
> - 推荐对照阅读：`docs/structures.md` / `docs/structures_zh.md`（架构），`docs/features.md` / `docs/features_zh.md`（功能）
>
> **小白补课入口**：如果你是推理小白，建议先从 `docs/background_zh/README.md` 按顺序阅读，再回到本文。
>
> **源码阅读入口（按顺序）**：如果你想“贴着源码逐文件逐函数读”，从 `docs/codewalk_zh/README.md` 开始。

---

## 目录

- 1. 项目定位与目标
- 2. 运行方式与“系统形态”
- 3. 代码结构总览（从仓库到包）
- 4. 进程/线程模型与通信（ZMQ + Torch/NCCL）
- 5. 关键抽象：Req / Batch / Context（推理时的“状态容器”）
- 6. 请求生命周期（从 API 到增量输出）
- 7. Scheduler：prefill、decode、chunked prefill、overlap scheduling
- 8. 内存模型与 KV Cache：paged KV、page table、token pool
- 9. Radix Cache：前缀复用、锁保护与逐出
- 10. Attention 后端：FlashAttention3 与 FlashInfer（为何要“混合后端”）
- 11. CUDA Graph：为什么能提升 decode 吞吐
- 12. Tensor Parallel（TP）：拆分、通信与 PyNCCL 加速
- 13. Kernel / JIT：自定义内核在系统中的角色
- 14. 开发者阅读路线与扩展点

---

## 1. 项目定位与目标

Mini-SGLang 是一个 **轻量但高性能** 的 LLM 推理/Serving 框架，目标是：

- **可读**：核心 Python 代码规模小，模块化明确，适合作为参考实现。
- **可跑**：提供 OpenAI-compatible API（`/v1/chat/completions`），支持在线服务与简单 benchmark。
- **可“现代”**：实现多个 Serving 关键优化：
  - **Radix Cache**：跨请求复用共享前缀的 KV cache
  - **Chunked Prefill**：长上下文预填充分块，降低峰值显存
  - **Overlap Scheduling**：CPU 调度/元数据处理与 GPU 计算重叠
  - **Tensor Parallelism**：多 GPU 扩展
  - **FlashAttention / FlashInfer**：高性能 attention 内核
  - **CUDA Graph**：降低 decode 阶段 CPU launch 开销

---

## 2. 运行方式与“系统形态”

最常见的使用方式是：

- **启动在线服务**：`python -m minisgl --model "Qwen/Qwen3-0.6B"`（示例见 `README.md`）
- **启动交互式 shell**：`python -m minisgl --shell`

从代码入口看：

- `python -m minisgl` → `python/minisgl/__main__.py` → `minisgl.server.launch_server()`
- 服务启动逻辑在 `python/minisgl/server/launch.py`

Mini-SGLang 在“在线服务”模式下是一个**多进程系统**：

- 1 个 **API Server**（FastAPI/uvicorn，负责 HTTP + streaming）
- N 个 **Tokenizer Worker**（把文本 → token ids）
- 1 个 **Detokenizer Worker**（把 token ids → 增量文本）
- TP 个 **Scheduler Worker**（每张 GPU 一个；rank0 负责与 tokenizer/detokenizer 打交道）

这套设计的核心动机是把复杂系统解耦成：**前端请求层 / 文本处理层 / GPU 调度计算层**。

---

## 3. 代码结构总览（从仓库到包）

### 3.1 仓库顶层目录

- `python/minisgl/`: 核心源码（Python）
- `python/minisgl/kernel/csrc/`: C++/CUDA 内核源码（AOT/JIT）
- `benchmark/`: offline/online benchmark 脚本
- `docs/`: 文档（英文为主；本文是中文总览）
- `tests/`: 单元测试

### 3.2 `minisgl` 包内部模块职责（开发者视角）

（这一节与 `docs/structures.md` 的“Code Organization”相呼应，但补充了更细的解释）

- `minisgl.server`
  - `api_server.py`: FastAPI 入口；负责将 HTTP 请求转成内部 ZMQ 消息，并以 SSE/streaming 返回结果
  - `launch.py`: 拉起后端多进程（TP schedulers + tokenizer/detokenizer）
  - `args.py`: CLI 参数与默认配置（端口、tp、cache、attn、cuda graph…）

- `minisgl.tokenizer`
  - `server.py`: tokenizer/detokenizer worker 的主循环（批量收消息，分别 tokenize 或 detokenize）
  - `tokenize.py` / `detokenize.py`: 具体实现（目前主要是逻辑清晰而非极致 tokenization throughput）

- `minisgl.scheduler`
  - `scheduler.py`: 主循环（prefill/decode 选择 + overlap scheduling）
  - `prefill.py`: pending 请求队列、chunked prefill 的切分与预算控制
  - `decode.py`: decode 阶段 running req 管理
  - `cache.py`: “页”分配/回收 + 对接 radix/naive cache manager
  - `io.py`: rank0 与 tokenizer/detokenizer 通信；TP 模式下向其它 ranks 广播消息
  - `table.py`: `token_pool`（token id 池）与 request table slot 管理

- `minisgl.engine`
  - `engine.py`: 单 GPU/单 rank 的执行引擎：加载模型、初始化 KV cache/attention、执行 forward、采样
  - `graph.py`: CUDA graph 捕获/回放（decode 加速关键）
  - `sample.py`: top-k/temperature 等采样逻辑

- `minisgl.attention`
  - `base.py`: attention backend 接口（metadata、capture/replay hooks）
  - `fa3.py`: FlashAttention3（通过 `sgl_kernel`）实现（prefill 常用）
  - `fi.py`: FlashInfer 实现（decode 常用，也可做 prefill）

- `minisgl.kvcache`
  - `mha_pool.py`: paged KV cache 的存储布局（按 layer/head/page）
  - `radix_manager.py`: radix tree 做前缀匹配与复用（核心）
  - `naive_manager.py`: 不复用前缀的 baseline

- `minisgl.distributed`
  - `impl.py`: torch.distributed 与 PyNCCL 两种通信后端的抽象（all-reduce/all-gather）
  - `info.py`: TP rank/size 信息

- `minisgl.kernel`
  - `store.py`: KV 写入内核（paged store）
  - `index.py`: embedding/weight indexing 类内核（高效 gather）
  - `radix.py`: 前缀比较内核（CPU AOT，快速比较 token ids）
  - `pynccl.py`: PyNCCL 封装（可选替代 torch distributed 通信）

---

## 4. 进程/线程模型与通信（ZMQ + Torch/NCCL）

**小白补课**：
- `docs/background_zh/11_zmq_msgpack_ipc.md`（为什么要多进程+消息队列）
- `docs/background_zh/10_tensor_parallel_nccl.md`（TP/NCCL 的最小知识）

### 4.1 进程是怎么拉起来的？

`minisgl/server/launch.py` 的 `launch_server()` 做了这些事：

1. 解析参数（模型、tp、端口、cache 类型…）
2. 设置多进程 start method 为 `spawn`（避免 CUDA fork 问题）
3. 拉起：
   - `world_size=tp` 个 scheduler 进程（每个 rank 一张 GPU）
   - 1 个 detokenizer 进程
   - `num_tokenizer` 个 tokenizer 进程
4. 等待子进程通过 `ack_queue` 报“ready”，再启动前端 API server

### 4.2 通信分两层：控制面 vs 数据面

Mini-SGLang 的通信可以理解为两类：

- **控制/小数据（跨进程消息）**：ZMQ + msgpack
  - API server ↔ tokenizer/detokenizer
  - tokenizer ↔ scheduler(rank0)
  - scheduler(rank0) ↔ scheduler(other ranks)（广播“同一条请求”到所有 rank）

- **大张量数据（跨 GPU/TP 通信）**：torch.distributed / NCCL（可选 PyNCCL）
  - 模型 TP 的 all-reduce / all-gather

### 4.3 ZMQ 消息与序列化

- ZMQ 队列封装在 `minisgl/utils/mp.py`：
  - PUSH/PULL（点对点队列）
  - PUB/SUB（广播）
  - 消息体用 `msgpack` 编码

- 消息类型在 `minisgl/message/`：
  - `frontend.py`: `UserReply`（增量输出 + finished）
  - `backend.py`: `UserMsg`（uid + input_ids + SamplingParams）、`ExitMsg` 等
  - `tokenizer.py`: `TokenizeMsg`、`DetokenizeMsg` 等
  - `utils.py`: 自定义序列化，支持把 1D CPU tensor 编进 bytes（当前限制：**只能序列化 1D tensor**）

### 4.4 TP 多 rank 时的“请求同步”策略（非常关键）

在 TP 模式下，需要保证 **所有 rank 看到完全相同的请求序列**，否则 TP 模型 forward 会死锁或输出不一致。

实现方式在 `minisgl/scheduler/io.py`：

- **rank0**：
  - 从 tokenizer pull 原始 bytes
  - 通过 ZMQ PUB 把原始 bytes 原封不动广播给其它 ranks
  - 用 `tp_cpu_group.broadcast()` 广播“这次一共有多少条 raw 消息”，让其它 ranks 知道要接收多少条

- **rank1..N-1**：
  - 从 ZMQ SUB 接收 rank0 广播的消息
  - 同样用 `tp_cpu_group.broadcast()` 接收消息条数，确保循环次数一致

这种做法优点是：**rank 间不需要解析/重编码消息**，广播 raw bytes 更轻。

---

## 5. 关键抽象：Req / Batch / Context（推理时的“状态容器”）

这三者是理解整个推理系统的钥匙（位于 `minisgl/core.py`）。

### 5.1 `Req`：一个请求在 GPU 推理中的状态机

`Req` 里最关键的字段：

- `host_ids`：CPU 上的 token ids（会随着 decode append）
- `cached_len`：已经“进入 KV cache / 已经算过”的长度（可理解为 prefix cache hit 后的起点）
- `device_len`：当前在 device 侧“已经可用”的长度（随每步 decode +1）
- `max_device_len`：输入 + 目标输出的总长度上限（由 `max_tokens` 决定）
- `table_idx`：该请求占用的 slot（索引到 `token_pool` / `page_table` 的行）
- `cache_handle`：KV cache manager 的句柄（radix/naive，用于锁与复用）

一些派生概念：

- `extend_len = device_len - cached_len`
  - prefill/extend 阶段需要真正计算的 token 数
- `remain_len = max_device_len - device_len`
  - decode 还剩多少 token 可以生成

### 5.2 `Batch`：一次 forward 的输入集合

`Batch` 有两个 phase：

- `prefill`: 处理新请求/扩展请求（可能一次吃多个 token）
- `decode`: 一步生成一个 token（每个 req `extend_len` 通常是 1）

batch 的核心字段：

- `reqs`: 当前真实请求列表
- `padded_reqs`: 为了 CUDA graph 或某些 kernel 对齐，可能会插入 dummy req 做 padding
- `input_ids`: 本轮 forward 实际输入 token（由 scheduler 从 `token_pool` gather）
- `out_loc`: 本轮新产生 KV 要写入的位置（页索引列表）
- `attn_metadata`: attention backend 为本 batch 准备的 metadata（如 cu_seqlens、page table、positions…）

### 5.3 `Context`：把“当前 batch”挂到全局上下文

很多模型层/attention 实现会从全局 context 读取当前 batch 的 metadata。

- `Context.forward_batch(batch)` 是一个 contextmanager：
  - 进入时 set 当前 batch
  - 退出时 reset
- 全局单例通过 `set_global_ctx/get_global_ctx` 管理

---

## 6. 请求生命周期（从 API 到增量输出）

下面以在线 `/v1/chat/completions` 为例，描述一条请求如何跑起来。

---

## 6.A 新手必看：端到端“时序”超细拆解（A）

> 这一节把“一条请求从 HTTP 到第一个 token 再到结束”的全链路按时间顺序拆成很多小步。你可以把它当作读源码时的“导航图”。
>
> 小白补课：
> - `docs/background_zh/01_llm_inference_overview.md`
> - `docs/background_zh/11_zmq_msgpack_ipc.md`
> - `docs/background_zh/08_cuda_basics_streams.md`

### 6.A.1 进程与通信通道（先定地图）

在线模式下的关键进程：

- **API Server**：`minisgl/server/api_server.py`
- **Tokenizer/Detokenizer Workers**：`minisgl/tokenizer/server.py`
- **Scheduler Workers（TP ranks）**：`minisgl/scheduler/scheduler.py`
- **Engine（每 rank 一个）**：`minisgl/engine/engine.py`

关键通信（控制面）：

- API → Tokenizer：`TokenizeMsg`
- Tokenizer → Scheduler(rank0)：`UserMsg`
- Scheduler(rank0) → Detokenizer：`DetokenizeMsg`
- Detokenizer → API：`UserReply`
- TP 模式下：Scheduler(rank0) → Scheduler(rank1..)：广播 raw msg（ZMQ PUB/SUB），并用 CPU 进程组广播“消息数量”

### 6.A.2 从 HTTP 到 token ids（Tokenizer 阶段）

以 `/v1/chat/completions` 为例（流式）：

1. **API Server 收到请求**（FastAPI handler）
2. 分配 `uid`（请求唯一标识）
3. 组装 `TokenizeMsg(uid, text, sampling_params)`，通过 ZMQ PUSH 发给 tokenizer
4. tokenizer worker 从 ZMQ PULL 收消息，分类：
   - `TokenizeMsg`：走 tokenize
   - `DetokenizeMsg`：走 detokenize
5. tokenize 将文本 → `input_ids`（CPU 1D int32 tensor）
6. tokenizer 把结果包装成 `UserMsg(uid, input_ids, sampling_params)` 发给 scheduler(rank0)

**你应该记住的关键点：**

- token ids 在进程间传输时会被序列化；Mini-SGLang 当前只支持序列化 **1D tensor**（见 `minisgl/message/utils.py`）。

### 6.A.3 Scheduler(rank0) 接入请求：准入与入队

7. Scheduler(rank0) 从 ZMQ 收到 `UserMsg`
8. 做一轮输入合法性检查：
   - `input_len < engine.max_seq_len`，否则丢弃
   - `max_tokens` 可能被调整，确保总长度不越界
9. 把请求转成 `PendingReq` 放入 `PrefillManager.pending_list`

### 6.A.4 TP 模式：为什么其它 ranks 也要“看到同一条请求”

10. 如果 `tp > 1`：
    - rank0 会把收到的 raw msg bytes 通过 ZMQ PUB 广播给 rank1..N-1
    - 同时用 CPU 进程组 broadcast 一次“这次有几条消息”，保证其它 rank 循环次数一致
11. rank1..N-1 订阅并解码同一条 `UserMsg`，也将其加入各自的 `pending_list`

**关键直觉：**

- TP 的 forward 需要多 rank 同步执行同一批次，否则很容易死锁或输出不一致。

### 6.A.5 Scheduler 主循环：选择 batch（prefill 优先）

12. Scheduler 进入主循环（正常或 overlap loop）
13. 调度策略（当前实现）：
    - 先尝试从 `pending_list` 取出一批做 prefill（带 token budget / chunked）
    - 若没有 runnable prefill，再把 `running_reqs` 做一个 decode batch

### 6.A.6 准备 batch：这一步决定“显存页”和“本轮输入”

14. 对于选出的 `Batch(reqs, phase)`，scheduler 会准备三类关键东西：

- **本轮要写 KV 的位置**：`out_loc`（page ids）
- **本轮 forward 的输入 token ids**：`batch.input_ids`
- **attention 的 metadata**：`batch.attn_metadata`（cu_seqlens/positions/page_table 等）

其中最关键的内存动作是：

- `CacheManager.allocate(needed_size)` 分配 page ids → `out_loc`
- 把 `out_loc` 写入 `page_table` 对应位置（这样 attention backend 才能查到 KV 在哪）

（B 部分会把每个张量的形状/索引详细展开）

### 6.A.7 Engine forward：计算 logits + 写 KV + 采样 next token

15. Scheduler 把 `batch.input_ids` 放到 Context（`with ctx.forward_batch(batch)`）
16. Engine 执行：
    - 如果满足条件：走 CUDA graph replay（decode 常见）
    - 否则：走普通 model.forward
17. attention backend 在 forward 中：
    - 把新 token 的 K/V 写入 KV cache（按 `out_loc` 的 page ids）
18. 生成 logits 后，Sampler 选出 `next_tokens_gpu`（GPU int32）
19. 发起 `next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)`
20. 记录一个 `copy_done_event`（用于之后精确同步）

### 6.A.8 处理输出：写回 token_pool + 回传 detokenize

21. Scheduler 把 `next_tokens_gpu` 写回 `token_pool` 的对应位置（作为后续输入来源）
22. 在 overlap loop 中，上一轮的 `copy_done_event` 会在“处理上一轮输出”时 `synchronize()`，再读取 CPU token
23. rank0 组装 `DetokenizeMsg(uid, next_token, finished)` 发给 detokenizer
24. detokenizer 生成增量文本，发 `UserReply(uid, incremental_output, finished)` 给 API
25. API 把增量文本通过 streaming 返回给用户

### 6.A.9 结束与资源回收（非常关键）

26. 如果某条请求 finished：
    - scheduler 从 decode running set 移除
    - 释放 `table_idx`（token_pool 的行）
    - 调用 `free_and_cache_finished_req(...)`：
      - 把 prefix（token ids → page ids）插入 radix cache（若启用）
      - 把可释放的页归还 free list
      - unlock 旧 handle（解除保护）

> C 部分会用“两个请求共享前缀”的例子，把 lock/unlock/insert/evict 的完整生命周期跑一遍。

### 6.1 API Server：接收请求并异步等待 token

文件：`minisgl/server/api_server.py`

核心逻辑：

1. HTTP 请求到达后分配一个 `uid`
2. 发送 `TokenizeMsg(uid, text, sampling_params)` 到 tokenizer（ZMQ）
3. API server 启动一个异步监听协程，不断从 tokenizer 侧接收 `UserReply`（增量文本）
4. 用 SSE/StreamingResponse 把增量输出写回客户端

### 6.2 Tokenizer Worker：分流 tokenization 与 detokenization

文件：`minisgl/tokenizer/server.py`

它在一个 while 循环中批量处理消息：

- 收到 `TokenizeMsg`：
  - 调 tokenizer 把 text → 1D CPU tensor（int32 token ids）
  - 组装成 `UserMsg(uid, input_ids, sampling_params)` 发给 scheduler(rank0)

- 收到 `DetokenizeMsg`：
  - 把 token id → string（增量）
  - 组装成 `UserReply(uid, incremental_output, finished)` 发给 API server

### 6.3 Scheduler：rank0 收请求并驱动 Engine 计算

文件：`minisgl/scheduler/scheduler.py`

rank0 收到 `UserMsg` 后：

- 校验长度（不能超过 `engine.max_seq_len`）
- 调整 `max_tokens` 避免越界
- 送入 `PrefillManager.pending_list`

随后 scheduler 的主循环会不断：

- 选择下一批要跑的 batch（prefill 优先，其次 decode）
- 准备 batch 的 `out_loc / input_ids / attn_metadata`
- 调用 `engine.forward_batch`
- 把 next_token 回到 detokenizer（rank0 才会发）

---

## 7. Scheduler：prefill、decode、chunked prefill、overlap scheduling

**小白补课**：
- `docs/background_zh/03_prefill_vs_decode.md`
- `docs/background_zh/06_batching_and_scheduling.md`
- `docs/background_zh/08_cuda_basics_streams.md`（理解 overlap 与 event/stream）

Scheduler 的关键目标是：**最大化 GPU 利用率，同时把 CPU 侧开销隐藏起来**。

---

## 7.B 新手必看：Scheduler 逐变量/张量形状拆解（B）

> 目标：把 scheduler 的关键张量/索引（`token_pool/page_table/out_loc/load_indices/write_indices/input_ids`）逐个解释清楚：**它是什么、形状是什么、谁写它、谁读它、什么时候同步**。
>
> 小白补课：
> - `docs/background_zh/05_paged_kv_and_page_table.md`
> - `docs/background_zh/08_cuda_basics_streams.md`

### 7.B.1 你需要先记住 3 个“存储面板”

Mini-SGLang 把“请求状态”拆成三块大内存面板（其中两块在 GPU）：

- **`token_pool`（GPU）**：存 token ids（int32）
  - 形状：`(max_running_req, max_seq_len)`（由 `TableManager` 分配）
  - 每条请求占用一行：`table_idx`

- **`page_table`（GPU）**：把 (table_idx, position) 映射到 KV 的 page id（int32）
  - 形状：`(max_running_req + 1, max_seq_len)`（+1 用于 dummy req）
  - 注意：当前实现 `page_size=1`，所以“一个 token 对应一个 page id”

- **KV cache buffer（GPU）**：真正存 K/V 的大 tensor（fp16/bf16）
  - 形状（概念上）：layers × pages × heads × head_dim × 2(K+V)
  - 由 `MHAKVCache` 持有

这三块的关系：

1. token_pool 提供本轮 forward 的输入 token ids
2. page_table 告诉 attention backend：这些 token 的 KV 在哪些 pages
3. KV cache buffer 存放 pages 对应的 K/V 数据

### 7.B.2 `table_idx`：每个请求的“行号”

每个请求会被分配一个 `table_idx`：

- 它同时用于：
  - 定位 `token_pool[table_idx, :]`
  - 定位 `page_table[table_idx, :]`

请求结束后会 `free(table_idx)`，释放该行给后续请求复用。

### 7.B.3 `cached_len` / `device_len`：两个“指针”分别表示什么

对于某个请求（`Req`）：

- `cached_len`：历史已经在 KV cache 中“可复用”的长度（前缀）
- `device_len`：当前已经“进入设备侧有效序列”的长度（会随 decode +1）

因此：

- `extend_len = device_len - cached_len`
  - 本轮如果做 prefill/extend，需要真正计算并写 KV 的 token 数

### 7.B.4 `needed_size` 与 `out_loc`：本轮要写 KV 的 page ids

当 scheduler 选出一个 batch 后，会计算：

- `needed_size = sum(req.extend_len for req in batch.reqs)`

然后调用：

- `batch.out_loc = cache_manager.allocate(needed_size)`

因此：

- `out_loc` 是一个 **GPU 1D int32 tensor**
- 长度 = 本 batch 里所有请求“本轮新增 token 数”之和

> 直觉：`out_loc` 就是“这一轮新增 token 的 KV 要写到哪些物理页”。

### 7.B.5 `load_indices`：把 token_pool 的 2D 切片压成 1D gather 索引

为了把每条请求本轮需要的 token ids 拼成一个连续的 1D 序列（供 attention backend 使用），scheduler 会构造：

- `load_indices`：GPU 1D int32 tensor

它来自一个辅助函数（示意）：

- 输入：若干段 `(table_idx, begin, end)`
- 输出：这些 2D 坐标在 `token_pool.view(-1)` 中的 1D index 列表

每条请求本轮需要加载的 token ids 范围是：

- `token_pool[table_idx, cached_len : device_len]`

因此 load_indices 对应的 ranges 是：

- `(req.table_idx, req.cached_len, req.device_len)`（对 padded_reqs 也会构造）

随后：

- `batch.input_ids = token_pool.view(-1)[load_indices]`

**你可以把它理解为：**

> 把“多行多段切片”变成“一维 gather”，这样后续 kernel 更好吃。

### 7.B.6 `write_indices`：把 next token 写回 token_pool 的位置

forward 结束后会产生每个请求的 `next_token`，需要写回 token_pool，作为下一轮输入的一部分。

因此 scheduler 构造：

- `write_indices`：GPU 1D int32 tensor

对应每条真实请求写一个位置：

- `token_pool[table_idx, device_len] = next_token`

所以 ranges 是：

- `(req.table_idx, req.device_len, req.device_len + 1)`

写入发生在：

- `token_pool.view(-1)[write_indices] = next_tokens_gpu`

### 7.B.7 写 `page_table`：把 out_loc 映射进页表（关键！）

注意：page_table 不是“自动更新”的。

Mini-SGLang 的做法是：复用 `load_indices` 的同一套 1D index，把它当作“页表中对应 token 位置”的索引：

- `page_table.view(-1)[load_indices] = batch.out_loc`

直觉解释：

- `load_indices` 指向的是“本轮新增 token 所在的 (table_idx, position)”集合
- `out_loc` 是这些 token 对应的 page ids
- 写进去后，attention backend 通过 `page_table` 就能找到这些 token 的 KV 应该写/读的页

### 7.B.8 `attn_metadata`：为什么每个 backend 都要 prepare？

attention backend 需要知道很多“形状与索引”信息才能执行 paged attention，例如：

- 每条样本的序列长度（seqlens）
- ragged layout 的 cu_seqlens
- positions（RoPE 位置）
- 对齐后的局部 page_table 或拼接后的 indices

因此 scheduler 在写完 page_table 后会调用：

- `engine.attn_backend.prepare_metadata(batch)`

不同 backend（FA3/FI）会生成不同结构的 metadata，但目的相同：**让 attention kernel 知道去哪里读 KV、如何组织 Q/K/V**。

### 7.B.9 overlap + event：什么时候需要同步？

Engine 会把 `next_tokens_gpu` 异步拷回 CPU，并返回一个 `copy_done_event`。

scheduler 在处理上一轮输出时会：

- `copy_done_event.synchronize()`：只在需要读取 CPU token 时同步

这比每步全局 `torch.cuda.synchronize()` 更细粒度，能降低不必要等待。

> 小白提示：如果你读到这里还不熟悉 stream/event/pinned memory，建议回到 `docs/background_zh/08_cuda_basics_streams.md`。

### 7.1 两个队列：pending（prefill）与 running（decode）

文件：`minisgl/scheduler/prefill.py` 与 `minisgl/scheduler/decode.py`

- `PrefillManager.pending_list`：等待进入系统的新请求
- `DecodeManager.running_reqs`：已经进入 decode 循环的请求集合

### 7.2 Prefill 的预算：`prefill_budget = max_extend_tokens`

prefill 不是无限塞：`PrefillManager.schedule_next_batch(prefill_budget)` 会用 token budget 控制：

- 每个请求只拿最多 `chunk_size = min(token_budget, remain_len)`
- 如果一个请求特别长，会变成 `ChunkedReq`，下次继续补剩余部分（chunked prefill）

这种设计的核心是避免“一个超长 prompt 把显存瞬间顶爆”。

### 7.3 overlap scheduling：双 stream + “上一轮结果处理”重叠

文件：`minisgl/scheduler/scheduler.py` 的 `overlap_loop`

核心思想：

- GPU forward 在 `engine.stream` 上跑（计算密集）
- scheduler 自己维护一个 `self.stream`（用于准备 indices/metadata 等“相对轻但可能阻塞”的工作）
- 每一轮：
  1. 先收消息并调度出本轮 batch
  2. 在 engine stream 上启动 forward
  3. 同时（在 CPU 侧）处理上一轮的输出 token、释放资源、发送 detokenize 消息

这样可以把 CPU 的“解包/调度/回收/发消息”等开销，尽可能埋到 GPU 计算时间里。

---

## 8. 内存模型与 KV Cache：paged KV、page table、token pool

**小白补课**：
- `docs/background_zh/04_kv_cache_basics.md`
- `docs/background_zh/05_paged_kv_and_page_table.md`

Serving 的核心瓶颈通常是：**KV cache 显存与带宽**。

Mini-SGLang 使用的核心数据结构可以这样理解：

### 8.1 `token_pool`：每个请求一行的 token id 存储

在 scheduler 中，token ids 会被复制到一个大池子（`TableManager.token_pool`）。

优势：

- batch gather 时能用连续内存 + indices gather
- 不必每轮都在 CPU 上拼接长序列传给 GPU

### 8.2 `page_table`：把“请求位置”映射到“KV 页索引”

`Engine` 会创建一个二维 int32 tensor：`page_table[(max_running_req + 1), max_seq_len]`。

- 行：请求 table slot（`table_idx`）
- 列：序列位置（token position）
- 值：对应 token 的 KV 页（page id）

注意：当前实现 `page_size=1`，也就是 **每个 token 对应一个 page id**（简化设计，便于讲清原理）。

### 8.3 `out_loc`：本轮要写 KV 的 page id 列表

每轮 forward 前，scheduler 会向 `CacheManager.allocate(needed_size)` 申请足够多的 pages，得到 `out_loc`。

随后 scheduler 会把它写进 `page_table` 的对应位置，这样 attention backend 就能用 page table 来查 KV。

### 8.4 KV cache 存储：`MHAKVCache`

文件：`minisgl/kvcache/mha_pool.py`

KV cache 的核心 buffer 是一个大 tensor（按 layer/head/page/head_dim 排列），并且会根据 TP 做 head 分片（每个 rank 只存 local heads）。

写入 KV 的关键是 `store_kv()`：

- 调 `minisgl.kernel.store_cache`，把 (k,v) 写入 paged KV 的指定 page ids

---

## 9. Radix Cache：前缀复用、锁保护与逐出

**小白补课**：
- `docs/background_zh/04_kv_cache_basics.md`（先理解 KV cache 是什么）
- `docs/background_zh/12_kernels_aot_jit.md`（理解 fast_compare_key 这类 kernel 的意义）

Radix cache 的目标是：当不同请求共享相同前缀时，**避免重复 prefill 计算**，直接复用已有 KV。

文件：`minisgl/kvcache/radix_manager.py`

---

## 9.C 新手必看：Radix + KV 生命周期“手把手跑一遍”（C）

> 目标：用一个具体例子解释：
>
> - 两个请求共享前缀时，Radix cache 如何命中
> - handle.lock/unlock 为什么必须做
> - 请求结束后 insert_prefix 是怎么让“前缀可复用”的
> - 显存不够时 evict 大概会淘汰谁（直觉 + 机制）
>
> 小白补课：
> - `docs/background_zh/04_kv_cache_basics.md`
> - `docs/background_zh/05_paged_kv_and_page_table.md`

### 9.C.1 场景设定：两个请求共享前缀

假设有两条请求：

- 请求 A：prompt token ids = `[1, 2, 3, 4, 5]`，需要生成若干 token
- 请求 B：prompt token ids = `[1, 2, 3, 9]`，需要生成若干 token

它们共享前缀 `[1,2,3]`。

系统希望做到：

- A 的 prefill 做过 `[1,2,3,4,5]` 后，B 到来时不要重复计算 `[1,2,3]`，直接复用 KV。

### 9.C.2 先理解“Radix cache 存什么”

Radix cache 的每个节点存两件事（概念上）：

- **key**：一段 token ids（压缩路径上的片段）
- **value**：这段 token ids 对应的 **page ids**（也就是 KV cache 的物理位置）

因此它本质上是一个映射：

> token 前缀 → KV 页索引（page ids）

### 9.C.3 请求 A 首次进入：match_prefix 命中 0

当 A 到来，cache 里还没有任何前缀：

1. scheduler 调 `cache_manager.match_req(A)`
2. radix manager 的 `match_prefix` 返回：
   - `prefix_len = 0`
   - `indices = []`
3. scheduler 继续正常 prefill：
   - 为 A 的新增 token 分配 `out_loc`（page ids）
   - 写入 `page_table`
   - attention forward 写入 KV cache

此时 A 已经在 GPU 上拥有了 `[1,2,3,4,5]` 对应的 KV 页。

### 9.C.4 A 结束（或阶段性完成）时：insert_prefix 把 KV 写进 radix tree

当 A finished，scheduler 会调用：

- `free_and_cache_finished_req(old_handle, input_ids, indices)`

其中关键动作是：

1. `insert_prefix(input_ids, indices)`：
   - 把 A 的 token ids 与 page ids 插入 radix tree
   - 这让未来的请求可以通过 token 前缀匹配拿到同一段 page ids

2. `unlock(old_handle)`：
   - 把 old_handle 对应路径上的 ref_count 减掉
   - 让这段前缀在无人使用时变为 evictable

> 直觉：A 结束后，它的 KV “不一定立刻释放”，而是被当成缓存前缀保留下来，等待复用或被逐出。

### 9.C.5 请求 B 到来：match_prefix 命中 `[1,2,3]`

B 到来时，radix tree 已经包含了 A 的前缀。

1. scheduler 调 `match_req(B)`（注意：实现里匹配的是 `input_ids[:input_len-1]`，避免把最后一个 token 当作可复用前缀）
2. radix manager 沿树走，调用 `fast_compare_key` 找最长匹配：
   - 返回 `prefix_len = 3`
   - 返回 `indices = [page(1), page(2), page(3)]`（示意）
3. scheduler 必须 `lock(handle)`：
   - 否则这些 indices 对应的页可能被并发逐出（evict）

此时 B 的 `cached_len=3`，意味着：

- `[1,2,3]` 这段不用再算
- B 只需要对剩余部分（`[9]` 以及后续生成 token）做 extend + decode

### 9.C.6 B 的 prefill/extend：只算“没命中的后缀”

对 B 来说：

- `extend_len = device_len - cached_len`

如果 B 的 prompt 是 `[1,2,3,9]`，那么初始：

- `device_len = 4`
- `cached_len = 3`
- `extend_len = 1`（只需要计算 token `9` 这一位的新增 KV）

因此本轮 `needed_size` 会更小，节省了：

- attention 计算
- KV 写入显存
- 时间与显存峰值

### 9.C.7 为什么 lock/unlock 必须严格成对？

因为 radix cache 允许 eviction 回收空间：

- `match_prefix` 返回的 indices 只是“当前看来存在的缓存页”
- 如果你不 lock，它们可能在你真正使用前被 `evict()` 回收并重新分配给别的请求

所以：

- 使用前 lock（让相关节点 ref_count>0 → protected）
- 使用完 unlock（ref_count 归零 → evictable）

### 9.C.8 evict 的直觉：回收“没人用且较旧”的前缀页

当系统需要新页（allocate 不够）时会触发 eviction。

Radix manager 维护：

- `evictable_size`：可逐出的缓存前缀页数量
- `protected_size`：被引用保护的页数量

并且节点上有 `timestamp`（被访问会更新）与 `ref_count`。

直觉上 eviction 会优先回收：

- `ref_count == 0`（无人使用）
- 且更“老”的前缀（较久没访问）

这样能最大化缓存命中率，同时保证正在服务的请求不被影响。

### 9.C.9 一个小结：Radix cache 在系统里到底省了什么？

在共享前缀场景下，它省掉的是：

- 重复的 prefill attention 计算
- 重复的 KV 写入与显存占用

因此它对“许多请求共享 system prompt / 相同前缀”的在线服务非常有效。

### 9.1 基本思路：用 radix tree 存 token 序列前缀

- 节点 key：一段 token ids（压缩路径）
- 节点 value：对应的 page indices（也就是 KV cache 的位置）
- 匹配：沿树走，找到最长匹配前缀

### 9.2 `match_prefix` 返回什么？

`match_prefix(input_ids)` 会返回：

- `handle`：句柄（含 prefix_len 与 node）
- `indices`：对应前缀的 page indices（拼起来的 1D tensor）

这里有一个非常重要的约束（在 `BaseCacheManager.match_prefix` 的注释里写得很清楚）：

> `match_prefix` 返回的 `indices` **只有在 handle 被 lock 后才安全可用**，否则可能被 eviction 回收。

### 9.3 “锁”与“逐出”：为什么需要 ref_count？

Radix cache 需要在 **“复用前缀”** 与 **“释放显存给新 token”** 之间做权衡，因此它把可用空间分两类：

- **protected_size**：被某些活跃请求引用、不能逐出的 prefix
- **evictable_size**：可以逐出的 prefix（一般是历史请求留下的、当前无人引用的前缀）

实现方式是对 radix tree 节点维护 `ref_count`：

- 当某个请求命中某段 prefix 并准备在 GPU 上使用它时，scheduler 会 `lock(handle)`：
  - 这会沿着该 node → root 的路径把 `ref_count` 加 1
  - `ref_count` 从 0 变 1 的节点就从 evictable 转为 protected

- 当请求结束并把自己的 prefix 插入 cache 后，会 `unlock(handle)`：
  - 沿路径把 `ref_count` 减 1
  - `ref_count` 变回 0 的节点就变成可逐出

你可以把它理解为：**Radix cache 把“KV cache 的一部分页面”变成一个带引用计数的共享资源**。

### 9.4 `insert_prefix`：把“已完成请求”变成未来可复用的缓存

当某个请求结束（或某段 prefix 完成）时，scheduler 会调用：

- `CacheManager.free_and_cache_finished_req(old_handle, input_ids, indices)`

它内部会：

1. 调 radix manager 的 `insert_prefix(input_ids, indices)`：把这段 (token ids → page ids) 写入 radix tree
2. 将“旧 handle 之外、但已经在 cache 中的那一段”的 page ids 归还到 free list（避免重复占用）
3. `unlock(old_handle)`：释放旧 handle 的保护引用

这样，一个请求的 KV 就能成为未来其它请求的“共享前缀缓存”。

### 9.5 `fast_compare_key`：为什么 prefix 匹配需要自定义内核？

radix tree 的核心操作是比较两段 token ids 的最长公共前缀长度。

Mini-SGLang 用了一个 AOT 编译的 C++ 实现：

- `minisgl/kernel/radix.py` → `load_aot("radix", cpp_files=["radix.cpp"])`
- `fast_compare_key(x, y)` 返回第一个不相等的位置（也就是 match_len）

这样做的意义在于：**prefix 比较是高频操作**，纯 Python/逐元素比较会拖慢调度。

---

## 10. Attention 后端：FlashAttention3 与 FlashInfer（为何要“混合后端”）

**小白补课**：
- `docs/background_zh/02_transformer_attention_basics.md`
- `docs/background_zh/03_prefill_vs_decode.md`（为什么 prefill/decode 适合不同 kernel）

文件：`minisgl/attention/base.py`、`fa3.py`、`fi.py`

### 10.1 为什么要区分 prefill 与 decode？

LLM 推理通常分两段：

- **prefill**：对输入 prompt 做一次（或几次 chunk）前向，计算整段序列的 KV
  - 计算量大（O(seq_len²) 注意力部分最重）
  - batch 形态常是“变长、多 token”

- **decode**：每步只生成 1 个 token（自回归）
  - 每步计算量相对小，但步数多
  - 启动/调度开销、kernel launch overhead 变得显著

因此最优的 attention kernel 往往不同：prefill 更看重吞吐，decode 更看重“每步固定形态的极致低开销”。

### 10.2 抽象接口：`BaseAttnBackend`

一个 backend 需要实现：

- `prepare_metadata(batch)`：为本 batch 生成 attention 所需的 metadata（例如 cu_seqlens、positions、page table…）
- `forward(q,k,v,layer_id,batch)`：执行 attention，并把 KV 写入 cache
- CUDA graph 相关：
  - `init_capture_graph(max_seq_len, bs_list)`
  - `prepare_for_capture(batch)`
  - `prepare_for_replay(batch)`

### 10.3 FlashAttention3（FA3）后端：典型用于 prefill

文件：`minisgl/attention/fa3.py`

要点：

- 每层 attention forward 都会先 `kvcache.store_kv(k, v, batch.out_loc, layer_id)`
  - 也就是把新 token 的 KV 写到 paged KV cache 的 page ids 上

- metadata 构造：
  - `cu_seqlens_k`：每个样本的 key 序列累计长度
  - `cu_seqlens_q`：query 的累计长度（prefill 可能与 K 不同，尤其有 cache hit 时）
  - `cache_seqlens`：每条样本的当前有效长度（用于 paged KV）
  - `page_table`：对齐到本 batch 的局部 page table
  - `positions`：由 `make_positions` 生成

调用底层 kernel：`torch.ops.sgl_kernel.fwd.default`（来自 `sgl_kernel.flash_attn`）

### 10.4 FlashInfer（FI）后端：典型用于 decode + CUDA graph

文件：`minisgl/attention/fi.py`

要点：

- FlashInfer 的 wrappers 会 `plan(...)` 一次，把 batch 形态（indptr/indices/last_page_len 等）固定下来
- `FIMetadata` 同时保留 CPU 与 GPU 的 cu_seqlens（FI API 需要）
- decode 场景下 `indices` 通常是 ragged/拼接的一维 page ids（`torch.cat([...])`）
- 有专门的 `CUDAGraphBatchDecodeWithPagedKVCacheWrapper` 用于 CUDA graph replay

注意：代码中明确写了一个经验性选择：

- flashinfer 的 fa3 目前有问题，prefill wrapper 使用 `backend="fa2"`

### 10.5 “混合后端”是怎么实现的？

`HybridBackend`（`attention/base.py`）会根据 `batch.is_prefill` 选择不同 backend：

- prefill → prefill_backend
- decode → decode_backend

这就是 `--attn fa3,fi` 这类参数背后的机制。

---

## 11. CUDA Graph：为什么能提升 decode 吞吐

**小白补课**：
- `docs/background_zh/08_cuda_basics_streams.md`
- `docs/background_zh/09_cuda_graph_basics.md`

文件：`minisgl/engine/graph.py`

### 11.1 decode 的一个常见瓶颈：CPU launch 开销

decode 是“一步一个 token”，如果每步都要：

- Python 调度
- 构造 metadata
- 触发一堆小 kernel

CPU 会成为瓶颈，GPU utilization 下降。

### 11.2 CUDA graph 的思路

把 decode 过程中的 kernel launch 序列“录制”为一个图（CUDAGraph），之后每步只需要：

- 更新少量输入 buffer（token ids / page ids / seqlens 等）
- `graph.replay()` 一下

这样能大幅降低每 token 的 CPU overhead。

### 11.3 Mini-SGLang 的实现要点

`GraphRunner` 会：

- 在启动时按一组 batch size（`cuda_graph_bs`）捕获多份 graph
- 把 logits buffer 预先分配好（最大 bs）
- 每次 decode：
  - 选择合适的 padded batch size
  - `attn_backend.prepare_for_replay(batch)`
  - `g.replay()`

为了让 graph 可复用，batch 往往需要 padding 到预捕获的大小（见 scheduler 的 `engine.graph_runner.pad_batch(batch)`）。

---

## 12. Tensor Parallel（TP）：拆分、通信与 PyNCCL 加速

**小白补课**：
- `docs/background_zh/10_tensor_parallel_nccl.md`

### 12.1 TP 进程模型

Mini-SGLang 的 TP 是“**一 GPU 一进程**”：

- rank i 绑定 `cuda:i`（见 `Engine.__init__`）
- 每个 rank 都运行自己的 scheduler + engine，但只有 rank0 对外发 detokenize（其它 ranks 只做计算）

### 12.2 TP 的通信抽象

文件：`minisgl/distributed/impl.py`

核心接口：

- `all_reduce(x)`
- `all_gather(x)`

默认使用 `torch.distributed`。

可选：`enable_pynccl_distributed(...)` 把 PyNCCL 插件压到栈顶，之后 `DistributedCommunicator` 会用 PyNCCL 实现。

### 12.3 PyNCCL 是什么？为什么要它？

文件：`minisgl/kernel/pynccl.py`

PyNCCL 通过 tvm-ffi 封装了一个 NCCL communicator，并提供：

- `all_reduce`
- `all_gather`
- 以及（可选）预分配的通信 buffer（减少频繁申请/释放）

它的价值通常是：

- 更细粒度地控制通信 buffer 与性能
- 在某些场景下能比纯 torch.distributed 更快/更稳定

---

## 13. Kernel / JIT：自定义内核在系统中的角色

**小白补课**：
- `docs/background_zh/12_kernels_aot_jit.md`

Mini-SGLang 的一个重要特点是：在关键路径上，它用少量自定义 kernel 把性能拉起来，但保持整体可读。

### 13.1 两种编译方式：AOT vs JIT

文件：`minisgl/kernel/utils.py`

- **AOT**（ahead-of-time）：
  - `load_aot(...)` 编译固定的 C++/CUDA 源码
  - 例：`radix.cpp`（prefix compare）、`pynccl.cu`（NCCL wrapper）

- **JIT**（just-in-time / template specialization）：
  - `load_jit(...)` 把 CUDA 源码 include 进来，按模板参数编译出专用 kernel
  - 例：`store.cu`/`index.cu` 会根据 element_size 等生成不同实例

### 13.2 KV 写入：`store_cache`

文件：`minisgl/kernel/store.py` + `minisgl/kvcache/mha_pool.py`

每层 forward 时都会调用 `store_cache(k_cache, v_cache, indices, k, v)`：

- `indices` 就是本轮的 `out_loc`（page ids）
- kernel 把新 token 的 K/V 写到 paged KV cache 对应页

这一步是 decode 性能的关键之一：写入必须足够快、并且尽量避免额外拷贝。

### 13.3 Weight indexing：`indexing`

文件：`minisgl/kernel/index.py`

常见用途是 embedding / 大矩阵按 token ids gather（尤其 vocab 很大时），通过 JIT specialization 提升吞吐。

---

## 14. 开发者阅读路线与扩展点

### 14.1 推荐阅读路线（从“能跑”到“能改”）

1. **端到端入口**
   - `minisgl/__main__.py`
   - `minisgl/server/launch.py`
   - `minisgl/server/api_server.py`

2. **消息与序列化**
   - `minisgl/utils/mp.py`
   - `minisgl/message/*`
   - `minisgl/tokenizer/server.py`

3. **调度与执行**
   - `minisgl/scheduler/scheduler.py`
   - `minisgl/engine/engine.py`

4. **性能关键**
   - `minisgl/scheduler/prefill.py`（chunked prefill）
   - `minisgl/kvcache/radix_manager.py`（prefix reuse）
   - `minisgl/engine/graph.py`（CUDA graph）
   - `minisgl/attention/fa3.py` 与 `fi.py`（backend）
   - `minisgl/kernel/*`（store/index/radix/pynccl）

### 14.2 常见扩展点（你可能会想改的地方）

- **调度策略**
  - 目前 `_schedule_next_batch` 是“prefill 优先”
  - 可以尝试 decode-first / aging / fairness / priority 等策略

- **缓存策略**
  - radix 的 eviction 策略（按 timestamp、ref_count 等）可以继续优化
  - page_size=1 是教学友好配置；真实系统会用更大的 page 以提升 locality

- **采样策略**
  - `SamplingParams` 目前很简化（top_k、temperature、max_tokens…）
  - 可以加入 top_p、repetition penalty、logprobs 等

- **模型支持**
  - 当前实现集中在 Llama/Qwen3；可以参考 `minisgl/models/` 扩展更多架构

- **通信后端**
  - torch.distributed ↔ PyNCCL 的切换与 buffer 策略

---

## 附：几个“理解系统的抓手”（总结）

如果你只记住几件事来理解 Mini-SGLang，建议是：

1. **`token_pool` 存 token ids，`page_table` 映射 token→KV 页，`out_loc` 是本轮新 KV 的写入位置**
2. **prefill 是“多 token/变长”，decode 是“一 token/固定形态”，所以 attention backend 与 CUDA graph 的策略不同**
3. **Radix cache 复用的是“前缀的 KV 页”，lock/unlock 解决“复用 vs 逐出”的并发安全**
4. **TP 模式下 rank0 广播 raw 消息保证所有 rank 调度一致；大张量通信走 NCCL**


