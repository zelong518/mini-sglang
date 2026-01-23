# 源码阅读路线（中文 · 按顺序从 0 到能改）

> 这是一套“贴近源码”的导读文档：每一篇都会告诉你**先看哪些文件、看哪些类/函数、关键变量是什么、建议从哪里打断点/打印**。
>
> 配套：
> - 总览：`docs/overview_zh.md`
> - 小白补课：`docs/background_zh/README.md`

---

## 建议阅读顺序

### 0. 先把系统跑起来（可选）

如果你还没跑过项目，建议先按 `README.md` 启动一次（单卡/小模型），再开始看代码，直觉会好很多。

---

## 01｜入口与启动链路（你从哪里进来？进程怎么拉起来？）

- `01_entry_and_launch.md`

你会看懂：

- `python -m minisgl` 最终调用了谁
- 后端子进程（TP schedulers + tokenizer/detokenizer）如何启动与 ready 同步
- API server 如何把请求塞进内部队列

---

## 02｜消息与序列化（进程间到底传什么？）

- `02_messages_and_ipc.md`

你会看懂：

- `TokenizeMsg/UserMsg/DetokenizeMsg/UserReply` 的字段与方向
- msgpack + 自定义序列化如何把 1D tensor 传过 ZMQ

---

## 03｜Tokenizer / Detokenizer Worker（文本↔token 的边界）

- `03_tokenizer_workers.md`

你会看懂：

- tokenizer worker 的主循环如何“混合处理 tokenization 和 detokenization”
- local batch（`local_bs`）的意义

---

## 04｜Scheduler 主循环（prefill/decode/overlap 的骨架）

- `04_scheduler_main_loop.md`

你会看懂：

- `overlap_loop` 的 pipeline 结构
- prefill 队列与 decode running set 的协作关系

---

## 05｜Batch 准备细节（最容易卡住的小白难点）

- `05_batch_preparation_deep_dive.md`

你会看懂：

- `token_pool/page_table/out_loc/load_indices/write_indices` 的构造与作用
- 为什么 `page_table.view(-1)[load_indices] = out_loc` 这句这么关键

---

## 06｜Engine forward + Sampler（一次 forward 到底做了什么？）

- `06_engine_forward_and_sampling.md`

你会看懂：

- model.forward / CUDA graph replay 的分支
- next token 的采样与 CPU/GPU 拷贝同步点（event）

---

## 07｜Attention backends（FA3/FI）与 metadata

- `07_attention_backends_metadata.md`

你会看懂：

- `prepare_metadata` 里到底在准备什么（cu_seqlens/page indices/positions）
- 为什么 decode 更适合某些 backend + CUDA graph

---

## 08｜KV cache 与 Radix cache（前缀复用的真实生命周期）

- `08_kvcache_and_radix_lifecycle.md`

你会看懂：

- `CacheManager.allocate/free_and_cache_finished_req`
- radix tree 的 match/insert/lock/unlock/evict 的关系

---

## 09｜CUDA Graph（GraphRunner 捕获与回放）

- `09_cuda_graph_runner.md`

你会看懂：

- capture 时需要准备哪些 buffer
- replay 时哪些数据会被 copy 到 capture buffer
- padding batch size 的原因

---

## 10｜TP/通信：数据面（NCCL）与控制面（rank0 广播）

- `10_tp_and_distributed.md`

你会看懂：

- `distributed/impl.py` 的 all-reduce/all-gather 抽象
- `kernel/pynccl.py` 的可选通信实现
- `scheduler/io.py` 如何让所有 rank 看到一致请求


