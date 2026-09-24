# CS336 Systems — Problem 4.2 `torch_compile`

## (a) Compiled Attention Benchmark

The assignment asks us to extend the previous PyTorch attention benchmark with a `torch.compile` version and compare compiled and uncompiled forward/backward timings under the same attention configurations.

The compiled run used 30 warm-up iterations before timing. The uncompiled numbers below are the measurements from the preceding `pytorch_attention` experiment.

| d_model | seq_len | Eager Fwd (ms) | Compiled Fwd (ms) | Fwd speedup | Eager Bwd (ms) | Compiled Bwd (ms) | Bwd speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.280 | 2.904 | 0.10× | 0.856 | 0.610 | 1.40× |
| 16 | 1024 | 1.447 | 6.942 | 0.21× | 3.384 | 1.497 | 2.26× |
| 16 | 4096 | 19.983 | 8.739 | 2.29× | 49.729 | 20.136 | 2.47× |
| 32 | 256 | 0.240 | 6.654 | 0.04× | 0.763 | 0.663 | 1.15× |
| 32 | 1024 | 1.363 | 2.203 | 0.62× | 3.427 | 1.665 | 2.06× |
| 32 | 4096 | 21.234 | 9.584 | 2.22× | 49.017 | 20.879 | 2.35× |
| 64 | 256 | 0.472 | 1.796 | 0.26× | 1.036 | 0.693 | 1.49× |
| 64 | 1024 | 1.336 | 2.269 | 0.59× | 3.450 | 1.722 | 2.00× |
| 64 | 4096 | 21.198 | 10.312 | 2.06× | 50.807 | 23.236 | 2.19× |
| 128 | 256 | 0.407 | 2.025 | 0.20× | 1.125 | 1.030 | 1.09× |
| 128 | 1024 | 1.869 | 2.786 | 0.67× | 4.599 | 2.705 | 1.70× |
| 128 | 4096 | — | 18.166 | — | — | 35.678 | — |

`d_model=128, seq_len=4096` has a compiled measurement, but the earlier eager run did not produce a corresponding timing, so no eager value or speedup is invented for that row. Larger requested configurations exceeded the available GPU-memory/run limit and are not assigned fabricated timings.

### Observations

The compiled backward pass is faster for every configuration for which both eager and compiled measurements are available. The improvement becomes especially clear for longer sequences. For example, at `d_model=16, seq_len=4096`, backward decreases from `49.729 ms` to `20.136 ms` (about `2.47×` faster).

Forward behavior is more mixed. At small sequence lengths, the measured compiled forward is slower than eager; for example, `d_model=16, seq_len=256` changes from `0.280 ms` to `2.904 ms`. At sequence length 4096, however, compilation gives a substantial improvement in the configurations with comparable eager results: `d_model=16` improves from `19.983 ms` to `8.739 ms` (about `2.29×`), `d_model=32` from `21.234 ms` to `9.584 ms` (about `2.22×`), and `d_model=64` from `21.198 ms` to `10.312 ms` (about `2.06×`).

The compiled run reported warnings that TensorFloat32 tensor cores were available but not enabled and that there were not enough SMs for the `max_autotune_gemm` mode. The benchmark settings were left unchanged rather than enabling a different FP32 matmul mode only for the compiled experiment, so the comparison remains based on the settings actually used for the two runs.

## (b) Compile the Entire Transformer Model

**Skipped / not completed at the user's request.**

The assignment asks for compiling the entire Transformer in the end-to-end benchmark and comparing vanilla versus compiled Transformer forward performance and the combined forward/backward/optimizer-step performance. This experiment was intentionally skipped, so no end-to-end compiled-model measurements are fabricated.
