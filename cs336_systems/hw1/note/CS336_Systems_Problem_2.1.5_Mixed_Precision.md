# CS336 Systems - Problem 2.1.5 Mixed Precision

## Problem (mixed_precision_accumulation)

FP16 accumulation 会在每一次加法后将结果保存回
FP16，因此每一步都会产生舍入误差，经过大量累加后误差会不断增加。

FP16 input + FP32 accumulation 的区别在于：

    FP16 value
        |
        v
    convert to FP32
        |
        v
    FP32 accumulator

转换不能恢复已经丢失的信息，但是可以避免后续累加过程继续损失精度。

因此 GPU 常使用：

    FP16/BF16 multiplication
    +
    FP32 accumulation

------------------------------------------------------------------------

# Problem (benchmarking_mixed_precision)

## (a) autocast dtype 分析

使用：

``` python
torch.autocast(device_type="cuda", dtype=torch.float16)
```

结果：

  部分             dtype
  ---------------- ---------------
  模型参数         torch.float32
  fc1 输出         torch.float16
  LayerNorm 输出   torch.float32
  logits           torch.float16
  loss             torch.float16
  gradients        torch.float32

autocast 不会修改模型参数，而是根据操作类型选择计算精度。

Linear/GEMM 使用 FP16 加速；LayerNorm 保持 FP32 保证稳定。

------------------------------------------------------------------------

## (b) LayerNorm 为什么特殊处理

LayerNorm 需要计算 mean 和 variance，这些属于 reduction
操作，需要精确累加。

FP16 精度较低，可能导致：

-   mean 不准确
-   variance 不准确
-   normalization 结果偏移

因此 LayerNorm 通常保持 FP32。

BF16 比 FP16
数值范围更稳定，但归一化中的累加精度问题仍然存在，因此仍可能需要 FP32。

------------------------------------------------------------------------

## (c) Mixed Precision Benchmark

实验文件：

    benchmark_mixed_precision.py

实验比较：

-   FP32
-   FP16
-   BF16

设置：

-   batch size: 4
-   context length: 512
-   warmup steps: 5
-   measurement steps: 10

结果：

  Precision     Average Time         Std   Peak Memory
  ----------- -------------- ----------- -------------
  FP32              2.4183 s    0.0632 s       7.08 GB
  FP16             0.04947 s   0.00063 s       5.40 GB
  BF16             0.05033 s   0.00095 s       5.40 GB

------------------------------------------------------------------------

## Runtime 分析

FP16/BF16 相比 FP32 有明显加速。

原因：

Transformer 大量计算来自 GEMM：

-   Linear layers
-   Attention projection
-   Feed Forward Network

FP16/BF16 可以利用 Tensor Core 加速矩阵乘。

------------------------------------------------------------------------

## Memory 分析

FP32 使用 7.08 GB。

FP16/BF16 使用 5.40 GB。

原因：

FP32:

    4 bytes / element

FP16/BF16:

    2 bytes / element

因此降低了存储和数据移动压力。

------------------------------------------------------------------------

## Nsight Kernel 分析

FP32 主要 kernel：

    magma_sgemmEx_kernel

FP16/BF16 主要 kernel：

    cutlass::Kernel2

说明低精度 GEMM 使用了优化计算路径。

------------------------------------------------------------------------

# 总结

Mixed Precision 的核心：

    低精度提高速度
    高精度保证稳定

实验验证：

-   FP16/BF16 显著降低运行时间。
-   FP16/BF16 减少显存占用。
-   Nsight 显示低精度使用优化 GEMM kernel。
