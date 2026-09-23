# CS336 Systems - Problem: nsys_profile

## (a) Forward Pass Profiling

使用 NVIDIA Nsight Systems 分析 Transformer forward。

运行命令：

``` bash
nsys profile \
--trace=cuda,cudnn,cublas,osrt,nvtx \
--pytorch=functions-trace,autograd-shapes-nvtx \
-- python benchmark1.py \
--model-size small \
--mode forward
```

使用 NVTX 标记：

-   forward
-   backward
-   optimizer

Forward 时间：

填写实际 Nsight 测量结果。

Nsight 和 benchmark 结果接近，差异来自 profiling 的额外开销。

------------------------------------------------------------------------

## (b) 最大 CUDA Kernel

Nsight 结果：

最大 kernel：

`cutlass::Kernel2`

占：

约 34.0%

另一个重要 kernel：

`magma_sgemmEx_kernel`

占：

约 21.1%

这些 kernel 属于矩阵乘法相关操作。

Transformer 中大量计算来自：

-   Attention projection
-   Feed Forward Network
-   LM Head

这些操作本质都是 GEMM（矩阵乘）。

------------------------------------------------------------------------

## (c) 其他重要 Kernel

  Kernel                          占比
  ------------------------------- -------
  vectorized_elementwise_kernel   19.4%
  elementwise_kernel              12.9%
  multi_tensor_apply_kernel       9.9%

分析：

Elementwise kernel 来源：

-   RMSNorm / LayerNorm
-   激活函数
-   residual connection
-   tensor 加减乘除

虽然 FLOPs 较低，但是需要大量显存访问，因此仍然消耗明显时间。

multi_tensor_apply_kernel 通常与 optimizer 参数更新有关。

------------------------------------------------------------------------

## (d) Training Step 与 Inference 对比

Inference：

只包含 forward。

主要时间来自：

-   Attention projection
-   Linear layers
-   GEMM

Training：

包含：

-   forward
-   backward
-   optimizer.step()

Backward 增加：

-   参数梯度计算
-   输入梯度计算

Optimizer 增加：

-   参数更新
-   momentum 更新
-   variance 更新

因此训练时：

矩阵乘仍然占主要部分，但由于增加 backward 和 optimizer
kernel，其比例下降。

------------------------------------------------------------------------

## (e) Attention 中 Softmax 与 Matmul 比较

在 Attention 中加入 NVTX：

-   QK matmul
-   softmax
-   PV matmul

实验结果：

  操作        时间
  ----------- -----------
  QK matmul   39.488 ms
  softmax     78.174 ms
  PV matmul   28.444 ms

总 Attention 时间：

146.106 ms

比例：

  操作        比例
  ----------- -------
  QK matmul   27.0%
  softmax     53.5%
  PV matmul   19.5%

分析：

理论上：

QK matmul：

O(S²d_k)

Softmax：

O(S²)

因此矩阵乘 FLOPs 更高。

但是实际运行中 softmax 更慢。

原因：

矩阵乘：

-   GPU 高度优化
-   CUDA GEMM
-   Tensor Core

Softmax：

属于 memory-bound 操作，需要：

-   读取 attention matrix
-   reduction
-   写回结果

因此性能不仅由 FLOPs 决定，也受到显存访问影响。

------------------------------------------------------------------------

## 总结

通过 Benchmark 和 Nsight Systems：

1.  Transformer 最大计算来源是矩阵乘（GEMM）。

2.  GPU 性能不仅取决于 FLOPs，还取决于：

-   memory access
-   kernel efficiency
-   GPU utilization

3.  Attention 中：

虽然 matmul FLOPs 大于 softmax，

但 softmax 可能成为实际性能瓶颈。

这说明后续 FlashAttention 优化的重要性。
