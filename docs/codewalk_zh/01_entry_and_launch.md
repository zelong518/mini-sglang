# 01｜入口与启动链路：从 `python -m minisgl` 到所有子进程

> 读完这一篇，你应该能把“系统怎么启动、启动了哪些进程、它们怎么连起来”的链路在脑子里串起来。

---

## 1. 你从哪里开始读？

建议按这个顺序点开文件：

1. `python/minisgl/__main__.py`
2. `python/minisgl/server/launch.py`
3. `python/minisgl/server/api_server.py`
4. `python/minisgl/server/args.py`

（可选）如果你还会用 shell 模式：

- `python/minisgl/shell.py`

---

## 2. 入口：`python -m minisgl` 实际调用了谁？

### 2.1 `__main__.py`

这里基本只有两行：

- `from .server import launch_server`
- `launch_server()`

所以真正的入口在 `minisgl.server.launch_server`。

---

## 3. 启动器：`launch_server` 做了哪几件事？

文件：`python/minisgl/server/launch.py`

你重点看两个函数：

- `_run_scheduler(args, ack_queue)`
- `launch_server(run_shell=False)`

### 3.1 `parse_args`：把 CLI 参数变成配置对象

`launch_server` 会调用 `parse_args(sys.argv[1:], run_shell)`。

你可以在 `python/minisgl/server/args.py` 看到：

- `ServerArgs`：包含大量 scheduler/engine/server 相关配置
- 各类地址（ZMQ ipc/tcp）如何拼出来

### 3.2 start_method=spawn：为什么不是 fork？

`start_subprocess()` 内有：

- `mp.set_start_method("spawn", force=True)`

直觉原因：CUDA 场景下 fork 非常容易把 CUDA context/线程状态带出问题；spawn 更安全（代价是启动慢一些）。

### 3.3 后端子进程：启动哪些？

`world_size = server_args.tp_info.size`

随后会启动：

- **TP 个 scheduler 进程**：每个 rank 一个（进程名形如 `minisgl-TP{i}-scheduler`）
- **1 个 detokenizer 进程**
- **num_tokenizers 个 tokenizer 进程**

### 3.4 ready 同步：为什么有 `ack_queue`？

启动后，主进程会阻塞等待子进程发“ready”：

- `for _ in range(num_tokenizers + 2): logger.info(ack_queue.get())`

这里的 `+2` 表示：

- 1 个 scheduler（rank0）ready
- 1 个 detokenizer ready
- N 个 tokenizer ready

这样可以保证：API server 启动时后端已经就绪，避免“前端收到请求但后端还没连好”。

---

## 4. Scheduler 进程入口：`_run_scheduler`

文件：`python/minisgl/server/launch.py`

重点看：

- `scheduler = Scheduler(args)`
- `scheduler.sync_all_ranks()`：TP 模式下先做一次 CPU barrier
- `if args.tp_info.is_primary(): ack_queue.put("Scheduler is ready")`
- `scheduler.run_forever()`

所以：

> **每个 scheduler 进程内部会创建一个 Engine，并进入永久循环**。

你下一篇读 scheduler 主循环会更顺（见 codewalk 04）。

---

## 5. 前端 API：什么时候启动？做什么？

`launch_server` 的最后一行是：

- `run_api_server(server_args, start_subprocess, run_shell=run_shell)`

文件：`python/minisgl/server/api_server.py`

你要抓住这个事实：

- API server 本身也会在生命周期里调用 `start_backend`（也就是上面的 `start_subprocess`）
- API server 通过 ZMQ 与 tokenizer 交互（异步队列），并把结果 SSE/stream 返回给用户

---

## 6. 建议断点/打印点（快速建立直觉）

如果你用 IDE 调试，建议在下面几个点打断点：

- `minisgl/server/launch.py::launch_server`：看 tp/port/addr 等最终配置
- `minisgl/server/launch.py::_run_scheduler`：看每个 rank 的 args 是否一致（除了 rank id）
- `minisgl/server/api_server.py::v1_completions`：请求如何转成 `TokenizeMsg`
- `minisgl/tokenizer/server.py::tokenize_worker`：tokenization/detokenization 如何分流

下一篇建议读：

- `02_messages_and_ipc.md`


