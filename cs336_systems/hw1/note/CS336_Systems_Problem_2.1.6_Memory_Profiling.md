# CS336 Systems — Problem 2.1.6 Memory Profiling

## Problem: `memory_profiling`

This write-up follows Section 2.1.6 of the assignment handout and records the results from the actual profiling runs.

## (a) Active Memory Timeline: Forward vs. Full Training Step

For the XL model with context length 128, the forward-only Active Memory Timeline stayed nearly flat at roughly 3.3 GiB, with only small temporary spikes. Since inference used `torch.no_grad()`, activations did not need to be retained for backward.

For a full training step, memory increased progressively during the forward pass as residuals / saved tensors accumulated for backward. It then reached a much larger peak around the backward / optimizer region before falling as saved tensors were released.

Observed timeline files:
- `xl_ctx128_forward_fp32.pickle`
- `xl_ctx128_train_fp32.pickle`

The forward-only timeline was approximately flat around 3.3 GiB. The full-training timeline climbed from roughly 9.8 GiB toward ~13 GiB and peaked near ~16 GiB.

## (b) Peak Memory Usage

| Context length | Forward-only peak | Full training-step peak |
|---:|---:|---:|
| 128 | 3.439 GiB | 16.740 GiB |
| 2048 | 5.831 GiB | OOM |

Increasing the context length from 128 to 2048 increased forward peak memory from 3.439 GiB to 5.831 GiB. The XL model with context length 2048 could not complete a full FP32 training step on the available GPU and ran out of memory.

## (c) Mixed-Precision Peak Memory

BF16 mixed precision was enabled using `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.

| Context length | Mode | FP32 peak | BF16 mixed-precision peak |
|---:|---|---:|---:|
| 128 | Forward | 3.439 GiB | 4.988 GiB |
| 128 | Full training step | 16.740 GiB | 16.713 GiB |
| 2048 | Forward | 5.831 GiB | 6.697 GiB |
| 2048 | Full training step | OOM | OOM |

In this setup, BF16 autocast did not significantly reduce peak memory during training: at context length 128, the full training-step peak changed only from 16.740 GiB to 16.713 GiB. The 2048-token full training step still ran out of memory.

## (d) Residual-Stream Activation Size

For the XL reference configuration:
- batch size = 4
- context length = 512
- `d_model = 2560`

One residual-stream activation therefore has shape:

```text
[4, 512, 2560]
```

The number of FP32 values is:

```text
4 × 512 × 2560 = 5,242,880
```

FP32 uses 4 bytes per element:

```text
5,242,880 × 4 = 20,971,520 bytes
```

Converting to MiB:

```text
20,971,520 / 1024^2 = 20 MiB
```

Therefore one FP32 residual-stream activation tensor is **20 MiB**.

## (e) Largest Allocation in the Forward-Pass Timeline

After reducing the Detail level in PyTorch `memory_viz`, the largest observed allocation was:

```text
512.0 MiB = 536,870,912 bytes
```

The relevant stack trace was:

```text
nn_utils.py:7: softmax
model.py:433: scaled_dot_product_attention
model.py:523: forward
```

The low-level operation was `THPVariable_div`, so the allocation came from a division operation inside the custom `softmax` implementation, called from `scaled_dot_product_attention`. Thus, the largest observed allocation originated from the attention softmax path.

## (f) Nsight Systems Memory Analysis

**Skipped / not completed at the user's request.**

The handout asks for an Nsight Systems analysis of memory saved for backward by a single `TransformerBlock`, the five largest contributing operations and their percentages, and an estimate of gradient-tensor memory during backward. This part was intentionally not completed, so no values are fabricated here.

A preliminary Nsight run with `--cuda-memory-usage=true` showed that a `TransformerBlock` increased memory by roughly:

```text
3.37 GiB -> 3.63 GiB
```

which is about 0.26 GiB (~266 MiB), with similar increases observed across several blocks. Because the requested Top-5 breakdown and backward gradient-memory analysis were not completed, this is recorded only as a preliminary observation.

## Summary

The profiling results show a clear difference between inference and training: forward-only execution had relatively flat memory usage, while training accumulated saved tensors through the forward pass and reached a much larger peak during backward / optimizer execution. Longer context increased memory use substantially, and the XL model at context length 2048 could not complete a full training step on the available GPU in either FP32 or BF16 autocast.
