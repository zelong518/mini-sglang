# 10｜Tensor Parallel（TP）与 NCCL：多 GPU 推理最小知识（面向小白）

> 目标：你不需要实现 TP，但要能看懂：
>
> - TP 把模型“切”成什么样
> - 为什么需要 all-reduce / all-gather
> - NCCL 在其中做什么
> - Mini-SGLang 的“多进程 TP”架构是什么

---

## 1. 为什么需要 TP？

当模型太大（参数、激活、KV cache）：

- 单卡显存放不下，或吞吐不够

这时你有几种典型方案：

- **数据并行（DP）**：多卡各自跑一份完整模型，分不同请求/样本
  - 适合训练，也可用于推理的“多实例”
  - 但如果单卡放不下模型，DP 无法解决

- **张量并行（TP / Tensor Parallel）**：把“同一层的矩阵/头”拆到多卡上
  - 每步推理需要多卡协同通信
  - 重点解决“单卡放不下/单卡算不过来”

- **流水并行（PP / Pipeline Parallel）**：把层按深度切到不同卡
  - 需要 pipeline 调度，推理延迟和实现复杂度更高

Mini-SGLang 主要实现的是 **TP**。

---

## 2. TP 到底把什么“切”开？

直觉版：把大矩阵的乘法拆开算。

在 Transformer 中，最常见的大矩阵来自：

- Attention 的 Q/K/V 投影、输出投影
- MLP 的两层线性（up/down）

TP 常见两种切法（你看到 all-reduce/all-gather 多半就对应它们）：

### 2.1 Column Parallel（按输出维度切）

把权重矩阵的“列”切到不同卡：

- 每张卡算出输出的一部分
- 最后把各卡的输出 **拼起来（all-gather）**

### 2.2 Row Parallel（按输入维度切）

把权重矩阵的“行”切到不同卡：

- 每张卡拿到输入的一部分
- 各自算出一个“部分和”
- 最后把部分和 **加起来（all-reduce）**

> 你不需要死记：记住“拼起来 = gather，加起来 = reduce”就够了。

---

## 3. NCCL 在其中做什么？

在多 GPU TP 中，通信主要发生在 GPU↔GPU（跨卡）：

- all-reduce（求和）
- all-gather（拼接）

NCCL 是 NVIDIA 提供的高性能 GPU 通信库，针对 NVLink/PCIe 等拓扑优化。

PyTorch 的 `torch.distributed`（NCCL backend）在底层就是调用 NCCL 来做这些操作。

---

## 4. “一 GPU 一进程”的 TP 架构（非常常见）

很多推理系统（包括 Mini-SGLang）采用：

- 每张 GPU 一个 OS 进程（process-per-GPU）
- 每个进程绑定自己的 `cuda:{rank}`
- 通过 NCCL 进程组做通信

优点：

- 简化 CUDA 上下文与资源管理
- 避免复杂的多线程 CUDA 同步问题

---

## 5. Mini-SGLang 中 TP 的落点

### 5.1 启动多进程

`python/minisgl/server/launch.py` 会按 `--tp` 启动 `world_size` 个 scheduler/engine 进程（每个 rank 一张 GPU）。

### 5.2 rank0 与其它 rank 的“控制面同步”

注意：除了 NCCL 的“大张量通信”，系统还要保证：

- 所有 ranks 的 scheduler 看到完全一致的请求序列

Mini-SGLang 采用：

- rank0 用 ZMQ PUB 把 raw 消息广播给其它 ranks
- 再用 CPU 进程组广播“消息数量”对齐循环次数

这属于“控制面同步”，和 NCCL 的“数据面通信”是两层不同东西。

### 5.3 all-reduce/all-gather 的抽象

在 `python/minisgl/distributed/impl.py`：

- `TorchDistributedImpl`：默认用 `torch.distributed` 实现 all-reduce/all-gather
- `PyNCCLDistributedImpl`：可选，用自定义 PyNCCL 封装实现通信

---

## 6. PyNCCL 是什么？（可选理解）

Mini-SGLang 提供了一个可选通信实现：

- `python/minisgl/kernel/pynccl.py`

它通过 tvm-ffi 封装 NCCL communicator，提供：

- `all_reduce`
- `all_gather`

你可以把它理解为：“绕过一部分 torch.distributed 的抽象，直接做更可控的 NCCL 调用”。

---

## 7. 你读源码时的一个抓手

当你看到某个模块/层里出现：

- `all_reduce(x)`：多半是 row-parallel 的部分和需要求和
- `all_gather(x)`：多半是 column-parallel 的输出需要拼接

下一篇建议读（理解 ZMQ/msgpack 进程解耦）：

- `11_zmq_msgpack_ipc.md`


