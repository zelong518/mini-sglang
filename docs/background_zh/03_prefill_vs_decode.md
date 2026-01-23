# 03｜prefill vs decode：两种阶段，两种瓶颈（面向推理小白）

> 目标：把“prefill/encode/TTFT”和“decode/generate/tokens/s”这些术语一次性讲清楚，并解释为什么推理系统的优化几乎总是分别对待它们。

---

## 1. 两个阶段的定义

### 1.1 prefill（也叫 encode）

输入：用户的 prompt token ids（长度可能很长）  
输出：最后一个位置的 logits + 把整段 prompt 的 KV 写入 KV cache

特点：

- 需要处理整个 prompt 的序列
- attention 在长序列上开销大
- 是用户看到第一个 token 之前的主要成本（影响 TTFT）

### 1.2 decode（也叫 generate）

输入：上一步生成的 token（或当前步需要生成的“新 token”）  
输出：下一个 token（重复 many steps）

特点：

- 每步只处理 1 个 token（或极短长度）
- 步数很多：生成 512 tokens 就要跑 512 步
- CPU overhead / kernel launch overhead 更显著（影响 tokens/s）

---

## 2. 为什么系统常见指标是 TTFT + tokens/s？

- **TTFT (time to first token)**：用户多久看到第一个 token
  - 大多由 prefill 决定

- **tokens/s（生成速度）**：生成阶段每秒多少 token
  - 大多由 decode 阶段决定

一个系统可能 TTFT 很快但 tokens/s 很慢（prefill 优化好，decode 没优化），也可能反过来。

---

## 3. 典型优化分别针对什么？

### 3.1 针对 prefill 的优化

- **chunked prefill**：把长 prompt 切块，避免峰值显存/OOM
- **更快的 attention kernel**：FlashAttention 等提升长序列吞吐
- **prefix cache / radix cache**：跨请求复用 shared prefix，减少重复 prefill

### 3.2 针对 decode 的优化

- **CUDA graph**：降低每步 CPU launch 开销
- **更适合 decode 的 attention backend**：例如 FlashInfer 的 decode wrapper
- **overlap scheduling**：让 CPU 的调度/拷贝/回收与 GPU 计算重叠

---

## 4. Mini-SGLang 是怎么区分的？

你会在 `python/minisgl/core.py` 看到：

- `Batch(phase="prefill" | "decode")`

调度器在 `python/minisgl/scheduler/scheduler.py` 中：

- 默认 **prefill 优先**（`prefill_manager.schedule_next_batch(...) or decode_manager.schedule_next_batch()`）

attention backend 在 `python/minisgl/attention/base.py` 中：

- `HybridBackend` 根据 batch phase 选择 prefill/decode backend

CUDA graph 在 `python/minisgl/engine/graph.py` 中：

- 只对 decode 的固定形态 batch 更有意义（可捕获、可复用）

---

## 5. 常见误区（小白容易踩）

- **误区 1：以为“decode 就是轻松”**
  - decode 单步轻，但步数多；系统 overhead 容易成为瓶颈

- **误区 2：以为“prefill 越大 batch 越好”**
  - 长 prompt 的 prefill 很吃显存，batch 过大容易 OOM

- **误区 3：把 KV cache 当成“免费的”**
  - KV cache 是显存大头，管理策略决定你能否稳定在线服务

下一篇建议读：

- `04_kv_cache_basics.md`


