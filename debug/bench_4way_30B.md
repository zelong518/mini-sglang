# 四种并行方式速度对比 — Qwen3-30B-A3B (MoE), 4×H200

测试条件：GPU 2-5（GPU 0-1 被他人 vLLM 占用，故用 4 卡而非 8 卡）；dummy 权重；
单请求（并发=1）；客户端流式测量；每请求唯一随机前缀防缓存；预热丢弃。
TP/EP 用标准 NCCL（--disable-pynccl）。所有方法 tp/ep/pp/cp = 4。

## Prefill 吞吐 (tok/s) — 越高越快

| 输入 | TP4 | EP4 | PP4 | CP4 |
|---|---|---|---|---|
| 8k   | **57,630** | 33,532 | 15,479 | 19,422 |
| 32k  | **55,573** | 41,457 | 23,561 | 22,310 |
| 128k | **29,814** | 26,408 |  9,530 |  9,037 |
| 256k | **17,874** | 16,776 |  5,226 |  4,994 |
| 512k |  **9,816** |  9,679 |  2,721 |  2,619 |

## Decode 吞吐 (tok/s) — 越高越快

| 输入 | TP4 (默认) | TP4 (关graph) | EP4 | PP4 | CP4 |
|---|---|---|---|---|---|
| 8k   | **206** | 31.3 | 15.4 | 140 | 27.6 |
| 32k  | **186** |  —   | 15.5 | 138 | 27.5 |
| 128k | 48.9 | 21.1 | 12.3 | **46.0** | 21.7 |
| 256k | 48.9 | 22.9 | 12.1 | 44.5 | 20.2 |
| 512k | 48.5 |  —   | 12.0 | 45.2 | 21.2 |

## TTFT (s) — 越低越好

| 输入 | TP4 | EP4 | PP4 | CP4 |
|---|---|---|---|---|
| 8k   | **0.14** | 0.24 | 0.52 | 0.41 |
| 128k | **4.29** | 4.85 | 13.4 | 14.2 |
| 512k | **52.2** | 52.9 | 188  | 195  |

## 关键结论

### 1. Prefill：TP 最优，EP 接近，PP/CP 慢 ~3×
TP/EP 把单请求的 prefill 真正并行到 4 卡。PP（stage 串行，无重叠）和 CP
（prefill 复制，每 rank 跑全量）对单请求都退化成 ~单卡，所以慢 ~3×。

### 2. Decode：CUDA graph 是决定性因素
- TP 默认 decode 206 tok/s（8k）领先，但**主要因为只有 TP 用了 CUDA graph**。
- 关掉 graph 后 TP 暴跌到 31.3（**6.6× 差距来自 graph**），因为 eager 模式下每层
  all-reduce 的 kernel launch 开销暴露。
- EP/PP/CP **都禁用了 CUDA graph**（decode 里有 all-to-all / P2P / all-gather
  集合通信，当前 graph 捕获不支持）。

### 3. 同为 eager 时（公平对比）：PP > TP ≈ CP > EP
- **PP 最快（140 vs TP-eager 31）**：整个 forward 只有 3 次 stage 间 P2P，而 TP
  有 48 次 per-layer all-reduce。eager 下集合通信**次数**主导延迟。
- **CP ≈ TP-eager**（128k: CP 21.7 vs TP-eager 21.1）：CP 每层 all-gather+merge
  的开销，恰好被"每 rank 只读 1/cp KV"的带宽收益抵消，与 TP eager 持平。
- **EP 最慢**：每个 MoE 层 all-to-all dispatch+combine（≥2 次集合通信/层）。

### 4. 各方法的真实定位
- **TP**：单请求延迟最优（prefill 并行 + decode 用 graph）。低并发长上下文首选。
- **EP**：prefill 与 TP 相当；单请求 decode 被 per-layer all-to-all 拖累。收益在
  **大 batch decode**（all-to-all 被 batch 摊薄），单请求不划算。
- **PP**：单请求 prefill 最慢（串行），但 eager decode 因集合通信少反而快。真实收益
  在**高并发吞吐**（流水线填满）。
- **CP**：当前是 decode KV-scan 分片的最小正确实现，eager decode ≈ TP。**但**：
  prefill 复制、KV 复制（无显存节省）、禁 graph。要兑现 CP 的长上下文价值需扩展为
  **sharded-KV 存储 + ring-attention prefill + graph 兼容的 decode 集合通信**。

### 5. 最大优化空间：让 EP/PP/CP 也能用 CUDA graph
TP 的 6.6× decode 优势几乎全来自 graph。若能把 NCCL 集合通信纳入 graph 捕获，
EP/PP/CP 的 decode 都能大幅提升。这是后续最高性价比的优化点。

> 注：受共享机器限制用了 4 卡/30B。本会话早前在 8 卡/235B 上测过 TP/EP/PP
> （见 bench_results.md），趋势一致。
