# CS336 Systems - Problem: benchmarking_script

## (a) Benchmark 脚本实现

我们实现了一个 Transformer 端到端性能测试脚本，用于测量模型运行时间。

支持：

-   small
-   medium
-   large
-   xl
-   10B

支持三种运行模式：

1.  Forward only

执行：

输入 token → Transformer forward → 输出 logits

2.  Forward + Backward

执行：

输入 token → Forward → Loss → Backward

3.  完整训练步骤

执行：

输入 token → Forward → Loss → Backward → optimizer.step() → zero_grad()

Benchmark 流程：

1.  初始化随机 Transformer 模型。
2.  创建随机 token batch。
3.  执行 warm-up iteration。
4.  使用 `torch.cuda.synchronize()` 等待 GPU 完成计算。
5.  测量多个 iteration 的时间。
6.  计算平均时间和标准差。

CUDA 默认异步执行，因此需要 synchronize，否则测量结果可能只是 CPU 提交
CUDA 任务的时间，而不是 GPU 实际计算时间。

------------------------------------------------------------------------

## (b) Benchmark 实验结果

实验设置：

-   warm-up steps: 5
-   measurement steps: 10

  模式                 平均时间       标准差
  -------------------- -------------- --------------
  Forward              填写实验结果   填写实验结果
  Forward + Backward   填写实验结果   填写实验结果
  完整训练步骤         填写实验结果   填写实验结果

分析：

Forward-only 只包含推理，因此耗时最低。

Forward + Backward 增加梯度计算，因此运行时间增加。

完整训练步骤增加
optimizer.step()，需要更新模型参数，因此耗时进一步增加。

Warm-up 后 GPU 执行进入稳定状态，测量结果更加稳定。

------------------------------------------------------------------------

## (c) Warm-up 对结果的影响

实验：

比较：

-   0 次 warm-up
-   1 次 warm-up
-   2 次 warm-up
-   5 次 warm-up

观察：

没有 warm-up 时：

-   平均时间更高
-   波动更明显

原因：

第一次 GPU 执行包含：

-   CUDA context 初始化
-   GPU memory 分配
-   CUDA kernel 初始化
-   PyTorch 内部准备

warm-up 可以去除这些额外开销，使 benchmark 更接近真实运行性能。
