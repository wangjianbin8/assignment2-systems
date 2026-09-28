```md
# DDP with Overlapping Individual Parameters 总结

## Minimal DDP 问题
- Naive DDP 在 backward 完成后才对所有 parameter.grad 做 all_reduce。
- 缺点：
  1. 每个参数单独通信，通信调用次数多。
  2. backward 和通信无法重叠，通信时间完全暴露。

## Overlapping Individual Parameter Gradients

核心思想：

> 当某个参数的 gradient 在 backward 中计算完成后，立即进行异步通信。

流程：

```
backward
   |
parameter grad ready
   |
register_post_accumulate_grad_hook
   |
async all_reduce(grad)
   |
继续 backward
```

## DDP 实现要点

1. 初始化：
- 使用 `dist.broadcast(param.data, src=0)` 同步所有 rank 的初始参数。

2. 注册 hook：
```python
param.register_post_accumulate_grad_hook(hook)
```

当 gradient ready：
- 调用 `dist.all_reduce(param.grad, async_op=True)`
- 保存返回的 communication handle。

3. optimizer.step() 前：
调用：

```python
finish_gradient_synchronization()
```
内部：
- `handle.wait()` 等待所有异步通信完成。
- 将 sum 后的 gradient 除以 `world_size` 得到平均梯度。

## 关键概念

- `all_reduce`：所有 rank 参与计算，并获得相同结果。
- `async_op=True`：立即返回 handle，通信可以与 backward 计算重叠。
- `register_post_accumulate_grad_hook`：gradient 累积完成后自动触发函数。

最终目标：

减少通信等待时间，让：

```
backward computation
        +
gradient communication
```

同时执行，提高 DDP 训练效率。