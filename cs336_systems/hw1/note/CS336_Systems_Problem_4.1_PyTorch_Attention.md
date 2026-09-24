# CS336 Systems — Problem 4.1 `pytorch_attention`

## Experimental setup

- Batch size: 8; no multi-head dimension.
- `d_model ∈ {16, 32, 64, 128}`.
- `seq_len ∈ {256, 1024, 4096, 8192, 16384}`.
- 100 forward and 100 backward passes, with warm-up and CUDA synchronization.
- Memory is measured after forward and before backward.

For `Q,K,V ∈ R^{B×N×D}`, naïve attention materializes attention-sized intermediates of shape `[B,N,N]`, so the dominant long-sequence memory term is quadratic in `N`.

## Benchmark results

| d_model | seq_len | Forward (ms) | Backward (ms) | Memory before backward (GiB) |
|---:|---:|---:|---:|---:|
| 16 | 256 | 0.280 | 0.856 | 0.012 |
| 16 | 1024 | 1.447 | 3.384 | 0.080 |
| 16 | 4096 | 19.983 | 49.729 | 1.024 |
| 32 | 256 | 0.240 | 0.763 | 0.021 |
| 32 | 1024 | 1.363 | 3.427 | 0.082 |
| 32 | 4096 | 21.234 | 49.017 | 1.032 |
| 64 | 256 | 0.472 | 1.036 | 0.022 |
| 64 | 1024 | 1.336 | 3.450 | 0.086 |
| 64 | 4096 | 21.198 | 50.807 | 1.047 |
| 128 | 256 | 0.407 | 1.125 | 0.024 |
| 128 | 1024 | 1.869 | 4.599 | 0.094 |

The remaining requested configurations exceeded the available GPU memory / could not be run within the available memory. No timings are fabricated for them.

## Observations

Sequence length has a much stronger effect on memory than `d_model`. At `d_model=16`, increasing sequence length from 1024 to 4096 increases measured memory before backward from `0.080 GiB` to `1.024 GiB`.

This follows from the scaling difference:

\[
Q,K,V: O(BND), \qquad \text{attention-sized intermediates}: O(BN^2).
\]

The timing shows the same long-sequence growth. For `d_model=16`, forward time grows from `1.447 ms` at `N=1024` to `19.983 ms` at `N=4096`, while backward grows from `3.384 ms` to `49.729 ms`.

## Memory accounting: `B=8, N=8192, D=16`

Each Q/K/V tensor contains

\[
8\times8192\times16=1,048,576
\]

FP32 elements, or

\[
1,048,576\times4=4\text{ MiB}.
\]

Thus Q+K+V require only about `12 MiB`.

A full attention-sized matrix has shape `[8,8192,8192]`:

\[
8\times8192^2=536,870,912
\]

FP32 elements, requiring

\[
536,870,912\times4
=2,147,483,648\text{ bytes}
=2\text{ GiB}.
\]

Therefore a **single** `[B,N,N]` FP32 intermediate already occupies about `2 GiB`. A softmax result of the same shape is another tensor of the same order, while training additionally needs saved tensors and temporary allocations.

For comparison, at `N=4096`, one `[8,4096,4096]` FP32 tensor is `0.5 GiB`. The measured pre-backward memory for `d_model=16, N=4096` was `1.024 GiB`, which is consistent in scale with multiple attention-sized tensors being live. This is a scale argument, not an exact decomposition of PyTorch autograd's saved tensors.

## Response

Memory saved for backward grows approximately quadratically with sequence length because naïve attention materializes tensors containing a `seq_len × seq_len` dimension. Doubling sequence length therefore makes the dominant attention-sized storage roughly four times larger. Increasing the embedding dimension primarily changes `O(BND)` tensors and has a much smaller memory effect once the quadratic attention matrices dominate.

To eliminate this quadratic materialization cost, attention can be computed in tiles instead of constructing the complete `N × N` score/probability matrix in GPU memory. This motivates FlashAttention-2: compute attention block-by-block and avoid explicitly materializing the full attention matrix.
