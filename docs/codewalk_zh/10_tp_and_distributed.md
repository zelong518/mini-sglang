# 10｜TP 与通信：数据面（all-reduce/all-gather）+ 控制面（rank0 广播请求）

> 目标：让你把“多 GPU 推理”拆成两件事来理解：
>
> - **数据面**：模型 TP 的张量通信（NCCL / torch.distributed / PyNCCL）
> - **控制面**：让所有 ranks 的 scheduler 看到一致请求序列（ZMQ 广播 + CPU barrier/broadcast）

---

## 1. 先打开哪些文件？

按这个顺序：

1. `python/minisgl/server/launch.py`（多进程：world_size=tp）
2. `python/minisgl/engine/engine.py`（`cuda:{rank}` 绑定 + init distributed）
3. `python/minisgl/distributed/impl.py`（all-reduce/all-gather 抽象）
4. `python/minisgl/kernel/pynccl.py`（可选：PyNCCL 通信实现）
5. `python/minisgl/scheduler/io.py`（控制面：rank0 广播请求）

补课：

- `docs/background_zh/10_tensor_parallel_nccl.md`
- `docs/background_zh/11_zmq_msgpack_ipc.md`

---

## 2. 先把“数据面”和“控制面”分清

### 2.1 数据面：模型 TP 的张量通信

发生在 GPU↔GPU：

- all-reduce（求和）
- all-gather（拼接）

通常由 NCCL 加速。

你在代码里看到的典型入口是：

- `DistributedCommunicator.all_reduce(x)`
- `DistributedCommunicator.all_gather(x)`

### 2.2 控制面：确保各 rank “做同一件事”

发生在 scheduler 的输入侧：

- rank0 收到 tokenizer 发来的请求消息
- rank0 必须把同一条请求广播给其它 ranks
- 否则：各 rank 的 forward 步骤不同步，TP 通信会死锁/错乱

---

## 3. 多进程 TP：为什么是一 GPU 一进程？

Mini-SGLang 的 TP 架构是：

- `tp` 张 GPU → `tp` 个 scheduler/engine 进程
- rank i 绑定到 `cuda:i`

这样可以：

- 避免多线程共享 CUDA context 的复杂同步问题
- 更清晰地管理每张卡的 stream、memory、通信资源

---

## 4. 数据面：`distributed/impl.py` 的读法

文件：`python/minisgl/distributed/impl.py`

你会看到一个“插件栈”式结构：

- 默认：`TorchDistributedImpl`（torch.distributed）
- 可选：`PyNCCLDistributedImpl`（PyNCCL）

`enable_pynccl_distributed(...)` 会把 PyNCCL 插件 append 到栈顶：

- 之后 all-reduce/all-gather 会走 PyNCCL

直觉：

- torch.distributed 更通用
- PyNCCL 可能在某些场景更可控/更高效（更贴近 NCCL）

---

## 5. PyNCCL：它到底做了什么？

文件：`python/minisgl/kernel/pynccl.py`

它通过 tvm-ffi 封装一个 NCCL communicator 对象：

- rank0 创建 nccl unique id，然后用 CPU 进程组广播给其它 ranks
- 各 rank 用同一个 id 初始化 communicator

之后提供：

- `all_reduce(input, op="sum")`
- `all_gather(output, input)`

你不需要深入 tvm-ffi 细节，但要记住：

- 这是一条“替代 torch.distributed 的通信实现”的路径

---

## 6. 控制面：`scheduler/io.py` 的读法（最关键）

文件：`python/minisgl/scheduler/io.py`

### 6.1 单卡：最简单

- rank0 从 tokenizer 的 ZMQ PULL 直接收消息

### 6.2 多卡：rank0 广播 raw bytes + broadcast 消息数量

rank0：

- `get_raw()` 收到 raw bytes
- `put_raw(raw)` 用 PUB 广播给其它 ranks
- `broadcast(len(pending_raw_msgs))` 告诉其它 ranks 这轮有几条

rank1..：

- 先接收 broadcast 的“条数”
- 再循环 `get()` 同样次数从 SUB 接收消息

这个设计的核心目的是：

> 让所有 ranks 的 scheduler 看到完全一致的请求序列，从而保证 TP forward 同步。

---

## 7. 建议断点/打印点

- `SchedulerIOMixin.__init__`：确认当前 rank 使用的是哪套 recv/send 实现
- `_recv_msg_multi_rank0`：打印 `len(pending_raw_msgs)` 与 broadcast 的值
- `_recv_msg_multi_rank1`：打印 `dst_length` 与实际接收次数
- `DistributedCommunicator.plugins`：确认是否启用了 PyNCCL


