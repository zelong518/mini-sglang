# 02｜消息与 IPC：进程间到底传什么？怎么序列化？（ZMQ + msgpack）

> 读完这一篇，你应该能回答：
>
> - API/tokenizer/scheduler/detokenizer 之间分别传什么消息？
> - 为什么需要自定义序列化？它的限制是什么？
> - TP 模式下 rank0 为什么要“广播 raw bytes”？

---

## 1. 建议先打开哪些文件？

按这个顺序看最顺：

1. `python/minisgl/utils/mp.py`（ZMQ 队列封装 + msgpack）
2. `python/minisgl/message/utils.py`（序列化/反序列化：对象↔dict↔bytes）
3. `python/minisgl/message/backend.py`（scheduler 侧收到的消息）
4. `python/minisgl/message/tokenizer.py`（tokenizer 收/发的消息类型）
5. `python/minisgl/message/frontend.py`（API 侧收到的回复）
6. `python/minisgl/tokenizer/server.py`（消息如何被分流处理）
7. `python/minisgl/scheduler/io.py`（TP 多 rank 消息同步）

---

## 2. ZMQ 队列封装：PUSH/PULL 与 PUB/SUB

文件：`python/minisgl/utils/mp.py`

你会看到 4 类常用封装：

- `ZmqPushQueue` / `ZmqPullQueue`：PUSH/PULL 模式（点对点队列）
- `ZmqPubQueue` / `ZmqSubQueue`：PUB/SUB 模式（广播）
- 以及对应的 async 版本（用于 FastAPI 的异步循环）

它们共同做的事情是：

1. 把 Python 对象通过 `encoder(obj)` → `dict`
2. `msgpack.packb(dict)` → bytes
3. `socket.send(bytes)`

接收时反过来：

1. `socket.recv()` → bytes
2. `msgpack.unpackb(bytes)` → dict
3. `decoder(dict)` → 具体消息对象

---

## 3. 消息类型分三类：frontend / tokenizer / backend

### 3.1 frontend：发回给 API 的用户可见输出

文件：`python/minisgl/message/frontend.py`

最核心的类型：

- `UserReply(uid, incremental_output, finished)`
  - `incremental_output`：增量文本（可能为空）
  - `finished`：是否结束

API server 会把它包装成 SSE/stream chunk 返回给客户端。

### 3.2 tokenizer：API ↔ tokenizer 之间的消息

文件：`python/minisgl/message/tokenizer.py`

你主要关心：

- `TokenizeMsg(uid, text, sampling_params)`：API → tokenizer（文本→token ids）
- `DetokenizeMsg(uid, next_token, finished)`：scheduler → detokenizer（token id→文本）

### 3.3 backend：tokenizer → scheduler 的消息

文件：`python/minisgl/message/backend.py`

核心类型：

- `UserMsg(uid, input_ids, sampling_params)`
  - `input_ids`：CPU 侧 1D tensor（token ids）

---

## 4. 自定义序列化：为什么不用“直接 pickle”？

文件：`python/minisgl/message/utils.py`

它的核心机制是：

- 所有对象序列化成 dict 时，会带一个 `__type__`
- 反序列化时用 `__type__` 去 `globals()` 查找类并构造对象

### 4.1 tensor 序列化的关键限制（非常重要）

当前实现里对 tensor 有强约束：

- **只能序列化 1D tensor**
- 通过 `tensor.numpy().tobytes()` 把 buffer 变成 bytes
- 反序列化时用 numpy dtype 重建，然后 `torch.from_numpy(...)`

这就是为什么消息里传的 `input_ids` 是 **1D**：更复杂的张量（比如 KV cache）不会走这个 IPC 频道，它们留在 GPU/进程内处理。

---

## 5. tokenizer worker：同一个进程里既做 tokenize 也做 detokenize

文件：`python/minisgl/tokenizer/server.py`

你会看到：

- 从一个 ZMQ PULL addr 收消息
- 把消息按类型分成两组：
  - `TokenizeMsg` → tokenize_manager → 生成 `UserMsg` → 发给 scheduler
  - `DetokenizeMsg` → detokenize_manager → 生成 `UserReply` → 发回 API

这个设计的直觉是：

- tokenize 与 detokenize 都依赖 tokenizer 模型（HF tokenizer），放一起更省事

---

## 6. TP 模式下的“控制面同步”：为什么广播 raw bytes？

文件：`python/minisgl/scheduler/io.py`

目标：**所有 ranks 必须看到完全一致的请求序列**。

做法（直觉版）：

- rank0 从 tokenizer 收到 bytes
- rank0 把 bytes 原封不动通过 PUB 广播给其它 ranks
- rank0 再用 CPU 进程组 broadcast 一个整数：这轮一共有多少条消息
- rank1..N-1 订阅并按次数接收，保证循环一致

为什么广播 raw bytes？

- rank 间少一次“解码→再编码”的开销
- 保证 byte-level 完全一致，减少边界 bug

---

## 7. 建议断点/打印点

- `minisgl/utils/mp.py::ZmqPushQueue.put`：看 encoder 后的 dict 大概长什么样
- `minisgl/message/utils.py::serialize_type`：看 `__type__` 与 tensor buffer 如何编码
- `minisgl/tokenizer/server.py` 的消息分流处：确认 TokenizeMsg/DetokenizeMsg 的路径
- `minisgl/scheduler/io.py::_recv_msg_multi_rank0/_recv_msg_multi_rank1`：看 TP 同步过程

下一篇建议读：

- `03_tokenizer_workers.md`


