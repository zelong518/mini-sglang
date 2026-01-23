# 11｜ZMQ + msgpack：为什么推理系统喜欢“消息队列 + 多进程”（面向小白）

> 目标：你能理解：
>
> - 为什么不把“API/Tokenizer/GPU 调度”写在一个进程里
> - PUSH/PULL 和 PUB/SUB 各自适合什么
> - msgpack 做了什么（序列化）
> - Mini-SGLang 的通信拓扑是怎样的

---

## 1. 为什么要多进程？

在线推理服务通常把系统拆成多个进程（或服务），常见原因：

- **隔离**：某个组件崩了不一定拖垮全部（更好排查/重启）
- **解耦**：文本处理（tokenizer）与 GPU 计算的资源/节奏不同
- **并行**：tokenizer 可以开多个 worker；GPU scheduler 每卡一个进程
- **避免 GIL/线程复杂度**：Python 多线程 + CUDA 同步很容易踩坑

Mini-SGLang 的拆分方式是：

- API server（HTTP）
- tokenizer/detokenizer workers（文本↔token）
- TP 个 scheduler/engine（每卡一个）

---

## 2. ZMQ 是什么（小白版）

ZMQ（ZeroMQ）可以理解为：

> 一个轻量的消息传递库，提供多种通信模式（队列、发布订阅等），让你在进程间传“消息”。

它不是 Kafka 那种持久化消息队列，更像“高性能 socket 模式库”。

---

## 3. PUSH/PULL：点对点队列（任务流）

PUSH/PULL 的直觉：

- PUSH：生产者往队列里塞消息
- PULL：消费者从队列里取消息

特点：

- 很适合“任务分发/结果回收”
- 可以多个 PULL 竞争消费（做负载均衡），也可以多个 PUSH 汇聚

Mini-SGLang 用 PUSH/PULL 做：

- API → tokenizer
- tokenizer → scheduler(rank0)
- scheduler(rank0) → detokenizer
- detokenizer → API

---

## 4. PUB/SUB：广播（同步所有 TP ranks）

PUB/SUB 的直觉：

- PUB：广播者发一条消息
- SUB：订阅者都能收到（类似广播）

Mini-SGLang 用 PUB/SUB 做：

- rank0 scheduler 把“同一条请求消息”广播给其它 TP ranks

这样保证所有 ranks 的调度输入一致（非常关键）。

---

## 5. msgpack：序列化（把对象变成 bytes）

进程间传的是 bytes，因此需要把“Python 对象”编码/解码。

Mini-SGLang 在 `python/minisgl/utils/mp.py` 里使用：

- `msgpack.packb(...)` 编码成 bytes
- `msgpack.unpackb(...)` 解码回 dict

再由 `python/minisgl/message/utils.py` 做“类型恢复”：

- 每条消息带一个 `__type__`
- 支持把 **1D CPU tensor** 编成 bytes（注意当前限制：只能 1D）

---

## 6. Mini-SGLang 的通信拓扑（你该怎么读）

建议你从这里看起：

- `python/minisgl/server/launch.py`：谁创建 socket、谁连接 socket
- `python/minisgl/utils/mp.py`：ZMQ 队列封装（PUSH/PULL/PUB/SUB）
- `python/minisgl/message/*`：消息类型与序列化
- `python/minisgl/scheduler/io.py`：TP 模式下 rank0 如何广播消息数量与内容

读懂通信之后，你再看 scheduler 主循环会顺很多。

下一篇建议读（理解 AOT/JIT kernel）：

- `12_kernels_aot_jit.md`


