import math
import gc
import torch
import triton

from cs336_systems.hw2.flash_forward_triton import FlashAttentionTriton
# ============================================================
# 1. Regular PyTorch Attention
#
# 不使用 PyTorch 自带的 FlashAttention。
# 明确写：
#
#   S = QK^T / sqrt(d)
#   P = softmax(S)
#   O = PV
#
# 并支持 causal masking。
# ============================================================
def pytorch_attention(
    Q,
    K,
    V,
    is_causal=True,
):
    """
    Regular PyTorch attention.

    Q: [B, Nq, D]
    K: [B, Nk, D]
    V: [B, Nk, D]

    return:
        O: [B, Nq, D]
    """

    d = Q.shape[-1]

    # --------------------------------------------------------
    # Attention scores:
    #
    # Q  : [B, Nq, D]
    # K^T: [B, D, Nk]
    #
    # S  : [B, Nq, Nk]
    # --------------------------------------------------------
    S = torch.matmul(
        Q,
        K.transpose(-2, -1),
    ) / math.sqrt(d)

    # --------------------------------------------------------
    # Causal masking
    #
    # query q 只能看到 key k <= q
    #
    # 所以 k > q 的位置要 mask。
    # --------------------------------------------------------
    if is_causal:
        Nq = Q.shape[-2]
        Nk = K.shape[-2]

        q_indices = torch.arange(
            Nq,
            device=Q.device,
        )

        k_indices = torch.arange(
            Nk,
            device=Q.device,
        )

        # shape: [Nq, Nk]
        causal_mask = (
            k_indices[None, :]
            > q_indices[:, None]
        )

        # causal_mask=True 的位置不能被 attention
        S = S.masked_fill(
            causal_mask,
            float("-inf"),
        )

    # --------------------------------------------------------
    # Softmax
    # --------------------------------------------------------
    P = torch.softmax(
        S,
        dim=-1,
    )

    # --------------------------------------------------------
    # Output
    #
    # P: [B, Nq, Nk]
    # V: [B, Nk, D]
    #
    # O: [B, Nq, D]
    # --------------------------------------------------------
    O = torch.matmul(
        P,
        V,
    )

    return O

# ============================================================
# 2. Benchmark 一个 configuration
# ============================================================
def benchmark_single_config(
    seq_len=1024,
    d_model=64,
    dtype=torch.bfloat16,
):
    """
    Benchmark:

        1. PyTorch Attention forward
        2. FlashAttention forward

        3. PyTorch Attention backward
        4. FlashAttention backward

        5. PyTorch forward + backward
        6. FlashAttention forward + backward
    """

    # PDF 要求 batch size = 1
    batch_size = 1

    device = "cuda"

    # ========================================================
    # 在 benchmark 开始之前生成输入。
    #
    # 不要把 torch.randn(...) 放进 do_bench，
    # 否则会把随机数生成时间也测进去。
    # ========================================================
    Q = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    K = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    V = torch.randn(
        batch_size,
        seq_len,
        d_model,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # upstream gradient:
    # dO = ∂L / ∂O
    dO = torch.randn_like(Q)

    # ========================================================
    # 3. Forward benchmark
    # ========================================================

    # --------------------------------------------------------
    # Regular PyTorch Attention forward
    # --------------------------------------------------------
    pytorch_forward_ms = triton.testing.do_bench(
        lambda: pytorch_attention(
            Q,
            K,
            V,
            is_causal=True,
        )
    )

    # --------------------------------------------------------
    # FlashAttention forward
    #
    # forward:
    #     Triton kernel
    # --------------------------------------------------------
    flash_forward_ms = triton.testing.do_bench(
        lambda: FlashAttentionTriton.apply(
            Q,
            K,
            V,
            True,  # is_causal
        )
    )

    # ========================================================
    # 4. Backward benchmark
    #
    # 这里只测 backward，
    # 所以 forward 先在 benchmark 外面计算好。
    # ========================================================

    O_pytorch = pytorch_attention(
        Q,
        K,
        V,
        is_causal=True,
    )

    O_flash = FlashAttentionTriton.apply(
        Q,
        K,
        V,
        True,
    )

    # --------------------------------------------------------
    # PyTorch backward
    #
    # retain_graph=True:
    # do_bench 会重复调用 backward。
    # 如果第一次就释放 graph，
    # 第二次就不能继续 backward。
    # --------------------------------------------------------
    pytorch_backward_ms = triton.testing.do_bench(
        lambda: torch.autograd.grad(
            outputs=O_pytorch,
            inputs=(Q, K, V),
            grad_outputs=dO,
            retain_graph=True,
        )
    )

    # --------------------------------------------------------
    # FlashAttention backward
    #
    # 当前作业要求的版本：
    #
    # forward  -> Triton
    # backward -> PyTorch + torch.compile
    # --------------------------------------------------------
    flash_backward_ms = triton.testing.do_bench(
        lambda: torch.autograd.grad(
            outputs=O_flash,
            inputs=(Q, K, V),
            grad_outputs=dO,
            retain_graph=True,
        )
    )


    # ========================================================
    # 5. End-to-End benchmark
    #
    # 这次 forward 也必须放进 benchmark，
    # 因为我们测的是：
    #
    #     forward + backward
    # ========================================================

    def pytorch_fwd_bwd():
        # Forward
        O = pytorch_attention(
            Q,
            K,
            V,
            is_causal=True,
        )

        # Backward
        torch.autograd.grad(
            outputs=O,
            inputs=(Q, K, V),
            grad_outputs=dO,
        )

    def flash_fwd_bwd():
        # Forward: Triton
        O = FlashAttentionTriton.apply(
            Q,
            K,
            V,
            True,
        )

        # Backward: PyTorch + torch.compile
        torch.autograd.grad(
            outputs=O,
            inputs=(Q, K, V),
            grad_outputs=dO,
        )

    pytorch_fwd_bwd_ms = triton.testing.do_bench(
        pytorch_fwd_bwd
    )

    flash_fwd_bwd_ms = triton.testing.do_bench(
        flash_fwd_bwd
    )

    # ========================================================
    # 6. 打印结果
    # ========================================================

    dtype_name = str(dtype).replace("torch.", "")

    print()
    print("=" * 80)

    print(
        f"seq_len={seq_len}, "
        f"d_model={d_model}, "
        f"dtype={dtype_name}"
    )

    print("=" * 80)

    print(
        f"{'Implementation':<20}"
        f"{'Forward(ms)':>15}"
        f"{'Backward(ms)':>15}"
        f"{'Fwd+Bwd(ms)':>15}"
    )

    print("-" * 80)

    print(
        f"{'PyTorch':<20}"
        f"{pytorch_forward_ms:>15.3f}"
        f"{pytorch_backward_ms:>15.3f}"
        f"{pytorch_fwd_bwd_ms:>15.3f}"
    )

    print(
        f"{'FlashAttention':<20}"
        f"{flash_forward_ms:>15.3f}"
        f"{flash_backward_ms:>15.3f}"
        f"{flash_fwd_bwd_ms:>15.3f}"
    )

    print("=" * 80)

    # 同时返回结果，后面做完整 sweep 时会方便很多
    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "dtype": dtype_name,

        "pytorch_forward_ms": pytorch_forward_ms,
        "pytorch_backward_ms": pytorch_backward_ms,
        "pytorch_fwd_bwd_ms": pytorch_fwd_bwd_ms,

        "flash_forward_ms": flash_forward_ms,
        "flash_backward_ms": flash_backward_ms,
        "flash_fwd_bwd_ms": flash_fwd_bwd_ms,
    }

import gc
import torch


def run_full_benchmark():
    # ========================================================
    # PDF 要求的参数 sweep
    # ========================================================

    seq_lens = [
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        # 16384,
        # 32768,
        # 65536,
    ]

    d_models = [
        16,
        32,
        64,
        128,
    ]

    dtypes = [
        torch.bfloat16,
        torch.float32,
    ]

    results = []

    total_configs = (
        len(seq_lens)
        * len(d_models)
        * len(dtypes)
    )

    current = 0

    # ========================================================
    # Cartesian product
    # ========================================================
    for dtype in dtypes:
        for d_model in d_models:
            for seq_len in seq_lens:
                current += 1

                dtype_name = str(dtype).replace(
                    "torch.",
                    "",
                )

                print()
                print(
                    f"[{current}/{total_configs}] "
                    f"seq_len={seq_len}, "
                    f"d_model={d_model}, "
                    f"dtype={dtype_name}"
                )

                try:
                    # ----------------------------------------
                    # 调用我们已经写好的单 configuration benchmark
                    # ----------------------------------------
                    result = benchmark_single_config(
                        seq_len=seq_len,
                        d_model=d_model,
                        dtype=dtype,
                    )

                    result["status"] = "OK"

                    results.append(result)

                except torch.OutOfMemoryError:
                    # ----------------------------------------
                    # CUDA OOM
                    # ----------------------------------------
                    print("OOM")

                    results.append({
                        "seq_len": seq_len,
                        "d_model": d_model,
                        "dtype": dtype_name,

                        "pytorch_forward_ms": None,
                        "pytorch_backward_ms": None,
                        "pytorch_fwd_bwd_ms": None,

                        "flash_forward_ms": None,
                        "flash_backward_ms": None,
                        "flash_fwd_bwd_ms": None,

                        "status": "OOM",
                    })

                except RuntimeError as e:
                    # ----------------------------------------
                    # 有些 CUDA OOM 可能表现为 RuntimeError
                    # ----------------------------------------
                    if "out of memory" in str(e).lower():
                        print("OOM")

                        results.append({
                            "seq_len": seq_len,
                            "d_model": d_model,
                            "dtype": dtype_name,

                            "pytorch_forward_ms": None,
                            "pytorch_backward_ms": None,
                            "pytorch_fwd_bwd_ms": None,

                            "flash_forward_ms": None,
                            "flash_backward_ms": None,
                            "flash_fwd_bwd_ms": None,

                            "status": "OOM",
                        })

                    else:
                        # 不是 OOM，就不要吞掉真正的 bug
                        raise

                finally:
                    # ----------------------------------------
                    # 每个 configuration 跑完都清理显存
                    # ----------------------------------------
                    gc.collect()
                    torch.cuda.empty_cache()

    return results

def print_results_table(results):
    print()
    print("=" * 145)

    print(
        f"{'seq_len':>8} "
        f"{'d_model':>8} "
        f"{'dtype':>10} "
        f"{'PT Fwd':>12} "
        f"{'PT Bwd':>12} "
        f"{'PT F+B':>12} "
        f"{'Flash Fwd':>12} "
        f"{'Flash Bwd':>12} "
        f"{'Flash F+B':>12} "
        f"{'status':>8}"
    )

    print("-" * 145)

    for r in results:

        # 如果某个 configuration OOM，
        # 就显示 "-"
        def fmt(x):
            if x is None:
                return "-"
            return f"{x:.3f}"

        print(
            f"{r['seq_len']:>8} "
            f"{r['d_model']:>8} "
            f"{r['dtype']:>10} "
            f"{fmt(r['pytorch_forward_ms']):>12} "
            f"{fmt(r['pytorch_backward_ms']):>12} "
            f"{fmt(r['pytorch_fwd_bwd_ms']):>12} "
            f"{fmt(r['flash_forward_ms']):>12} "
            f"{fmt(r['flash_backward_ms']):>12} "
            f"{fmt(r['flash_fwd_bwd_ms']):>12} "
            f"{r['status']:>8}"
        )

    print("=" * 145)

if __name__ == "__main__":

    results = run_full_benchmark()

    print_results_table(results)