# 03｜Tokenizer/Detokenizer Worker：文本↔token 的边界与批处理

> 读完这一篇，你应该能回答：
>
> - tokenizer worker 为什么要同时处理 tokenize 与 detokenize？
> - `local_bs` 是什么？它对吞吐有什么影响？
> - tokenizer 如何把 `TokenizeMsg` 变成 `UserMsg`，detokenizer 如何把 `DetokenizeMsg` 变成 `UserReply`？

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/tokenizer/server.py`（主循环）
2. `python/minisgl/tokenizer/tokenize.py`（TokenizeManager）
3. `python/minisgl/tokenizer/detokenize.py`（DetokenizeManager）
4. `python/minisgl/message/tokenizer.py`（TokenizeMsg/DetokenizeMsg）
5. `python/minisgl/message/backend.py`（UserMsg）
6. `python/minisgl/message/frontend.py`（UserReply）

---

## 2. 为什么要把 tokenize 和 detokenize 放在同一个 worker？

直觉原因：

- 两者都依赖同一个 HF tokenizer（加载成本不低）
- 在线服务里 detokenize 也是高频操作（每 token 一次），单独拆进程未必更省资源
- 放在一个循环里能自然做“批处理”（同时积累多条消息一起处理）

因此在 `tokenize_worker` 里你会看到：

- 同一个 `recv_listener` 收到的消息既可能是 `TokenizeMsg` 也可能是 `DetokenizeMsg`
- worker 会按类型分流处理

---

## 3. `tokenize_worker` 的主循环：最关键的几行该怎么读？

文件：`python/minisgl/tokenizer/server.py`

你抓住这几个点：

### 3.1 连接了哪三条管道？

- `recv_listener`：从 `addr` 收消息（来自 API 或 scheduler）
- `send_backend`：把 `UserMsg` 发给 scheduler(rank0)
- `send_frontend`：把 `UserReply` 发给 API

### 3.2 `_unwrap_msg`：为什么要支持 batch 消息？

系统里有 `BatchTokenizerMsg` 这种批消息类型，所以 worker 会把它“展开成列表”统一处理。

### 3.3 local batch：`local_bs` 的意义

主循环里会做：

- 先 `get()` 一条
- 然后只要队列里还有消息，并且当前累计条数 `< local_bs`，继续多取几条

这相当于在 tokenizer worker 内做一次“微型 batching”：

- 优点：减少 msgpack/ZMQ 的 per-message 开销，提高吞吐
- 代价：可能增加一点点排队等待（延迟）

如果你追求极致 TTFT（低延迟），可能会把 `local_bs` 设小；追求吞吐可以设大。

---

## 4. Tokenize 路径：`TokenizeMsg` → `UserMsg`

当 worker 收到 `TokenizeMsg(uid, text, sampling_params)`：

1. 调 `TokenizeManager.tokenize(...)` 得到 `input_ids`（CPU 1D int tensor）
2. 为每条消息构造 `UserMsg(uid, input_ids, sampling_params)`
3. 如果只有 1 条，则直接发单条；如果多条，打包成 `BatchBackendMsg`
4. 用 `send_backend.put(...)` 发给 scheduler(rank0)

你需要特别记住：

- `input_ids` 是 1D CPU tensor（原因见 codewalk 02：序列化限制）

---

## 5. Detokenize 路径：`DetokenizeMsg` → `UserReply`

当 worker 收到 `DetokenizeMsg(uid, next_token, finished)`：

1. 调 `DetokenizeManager.detokenize(...)` 得到增量字符串 `reply`
2. 构造 `UserReply(uid, incremental_output=reply, finished=finished)`
3. 发送给 API server（同样支持 batch）

小白常见疑问：

- “为什么 detokenize 也是一个 worker？”
  - 因为 decode 每步都会产生 token，需要高频把 token→字符串，并且要维护 tokenizer 的一些状态/策略

---

## 6. 建议断点/打印点

- 在 `tokenize_worker` 的分流处打印两类 msg 的数量（`len(tokenize_msg)` / `len(detokenize_msg)`）
- 在 `TokenizeManager.tokenize` 输出处打印 `len(input_ids)`（检查 prompt tokenization 长度）
- 在 `DetokenizeManager.detokenize` 输出处打印字符串片段（验证 streaming 是否正确）

下一篇建议读：

- `04_scheduler_main_loop.md`


