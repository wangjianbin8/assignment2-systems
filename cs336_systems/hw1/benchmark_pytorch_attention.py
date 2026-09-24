import math
import time
import gc

import torch

# ------------------------------------------------------------
# 改成你自己项目中 scaled_dot_product_attention 的真实 import 路径
# ------------------------------------------------------------
from cs336_basics.model import scaled_dot_product_attention


# ============================================================
# Assignment 要求的实验配置
# ============================================================

BATCH_SIZE = 8

# PDF 指定的 head embedding dimensions
DIMS = [16, 32, 64, 128]

# PDF 指定的 sequence lengths
SEQ_LENS = [
    256,
    1024,
    4096
]

# 每个配置测 100 次
NUM_TRIALS = 100

# warm-up 次数 PDF 没有指定具体数字。
# 这里选择 10 次作为实验实现上的选择。
NUM_WARMUP = 100


def clear_cuda():
    """
    尽量清理上一个实验 configuration 留下的对象和 CUDA cache。

    注意：
    empty_cache() 并不会删除仍然被 Python tensor 引用的显存，
    所以前面还需要让相关 tensor 离开作用域 / 被删除。
    """
    gc.collect()
    torch.cuda.empty_cache()


def make_inputs(seq_len, d_model):
    """
    创建题目要求的 Q, K, V。

    题目要求：
        batch_size = 8
        不使用 multi-head attention

    所以 shape 是：
        [batch_size, seq_len, d_model]

    因为后面需要 benchmark backward，
    所以 requires_grad=True。
    """

    Q = torch.randn(
        BATCH_SIZE,
        seq_len,
        d_model,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    K = torch.randn(
        BATCH_SIZE,
        seq_len,
        d_model,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    V = torch.randn(
        BATCH_SIZE,
        seq_len,
        d_model,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    return Q, K, V


def attention_forward(Q, K, V):
    """
    调用我们 Assignment 1 已经实现好的 attention。

    如果你的函数参数形式不完全一样，
    只需要修改这里即可。
    """
    compiled_attention = torch.compile(scaled_dot_product_attention)

    return compiled_attention(
        Q,
        K,
        V,
    )


def benchmark_one_config(seq_len, d_model):
    """
    Benchmark 一个 (sequence length, d_model) configuration。

    返回：
        forward_ms
        backward_ms
        memory_before_backward_gib
    """

    clear_cuda()

    # --------------------------------------------------------
    # 1. 创建 Q, K, V
    # --------------------------------------------------------

    Q, K, V = make_inputs(seq_len, d_model)

    # ========================================================
    # 2. Warm-up
    # ========================================================
    #
    # GPU 第一次执行某些操作可能包含额外初始化成本，
    # 所以正式计时之前先运行几次。
    #
    # warm-up 不需要保留 autograd graph，因此这里使用 no_grad()
    # 来避免不必要的 activation 保存。
    # ========================================================

    with torch.no_grad():
        for _ in range(NUM_WARMUP):
            output = attention_forward(Q, K, V)

            # PDF 明确要求 synchronize
            torch.cuda.synchronize()

    del output

    # ========================================================
    # 3. Benchmark forward
    # ========================================================

    forward_times = []

    for _ in range(NUM_TRIALS):

        # synchronize 以后才开始计时，
        # 确保前面的 CUDA 工作已经完成。
        torch.cuda.synchronize()

        start = time.perf_counter()

        output = attention_forward(Q, K, V)

        # CUDA 默认异步。
        # 如果这里不 synchronize，测到的主要可能只是 CPU
        # 提交 kernel 的时间。
        torch.cuda.synchronize()

        end = time.perf_counter()

        forward_times.append(end - start)

        # 这一轮 forward 的 graph 不再使用。
        # 及时释放 Python 引用。
        del output

    # 秒 -> 毫秒
    forward_ms = (
        sum(forward_times)
        / len(forward_times)
        * 1000
    )

    # ========================================================
    # 4. Measure memory before backward
    # ========================================================
    #
    # 这里重新做一次 forward。
    #
    # 这一次不能用 torch.no_grad()，
    # 因为我们就是希望 autograd 保存 backward 所需要的 tensor。
    # ========================================================

    output = attention_forward(Q, K, V)

    # 把输出变成 scalar，方便 backward。
    loss = output.sum()

    torch.cuda.synchronize()

    # 当前时刻：
    #
    # forward 已经完成
    # backward 还没有开始
    #
    # 因此这里可以观察 forward + saved tensors
    # 所占用的 CUDA memory。
    memory_before_backward_bytes = torch.cuda.memory_allocated()

    memory_before_backward_gib = (
        memory_before_backward_bytes / (1024 ** 3)
    )

    # 这个 graph 后面不需要真正拿来计时，
    # 因为每次 backward benchmark 都需要自己的新 graph。
    del loss
    del output

    # ========================================================
    # 5. Warm-up backward
    # ========================================================

    for _ in range(NUM_WARMUP):

        # 每次重新 forward，创建新的 autograd graph。
        output = attention_forward(Q, K, V)
        loss = output.sum()

        loss.backward()

        torch.cuda.synchronize()

        # PyTorch 默认会累加 gradient。
        # 下一轮开始前把它们清掉。
        Q.grad = None
        K.grad = None
        V.grad = None

        del loss
        del output

    # ========================================================
    # 6. Benchmark backward
    # ========================================================

    backward_times = []

    for _ in range(NUM_TRIALS):

        # ----------------------------------------------------
        # 先建立新的 autograd graph。
        #
        # 注意：
        # 这一部分不计入 backward timing。
        # ----------------------------------------------------

        output = attention_forward(Q, K, V)
        loss = output.sum()

        torch.cuda.synchronize()

        # ----------------------------------------------------
        # 这里只测 backward
        # ----------------------------------------------------

        start = time.perf_counter()

        loss.backward()

        torch.cuda.synchronize()

        end = time.perf_counter()

        backward_times.append(end - start)

        # ----------------------------------------------------
        # 清理 gradient 和 graph
        # ----------------------------------------------------

        Q.grad = None
        K.grad = None
        V.grad = None

        del loss
        del output

    backward_ms = (
        sum(backward_times)
        / len(backward_times)
        * 1000
    )

    # --------------------------------------------------------
    # 清理当前 configuration
    # --------------------------------------------------------

    del Q
    del K
    del V

    clear_cuda()

    return (
        forward_ms,
        backward_ms,
        memory_before_backward_gib,
    )


def main():

    print(
        f"{'d_model':>8} "
        f"{'seq_len':>10} "
        f"{'forward(ms)':>15} "
        f"{'backward(ms)':>15} "
        f"{'memory(GiB)':>15} "
        f"{'status':>10}"
    )

    print("-" * 82)

    # --------------------------------------------------------
    # PDF 要求遍历两个列表的 Cartesian product。
    # --------------------------------------------------------

    for d_model in DIMS:

        for seq_len in SEQ_LENS:

            try:

                (
                    forward_ms,
                    backward_ms,
                    memory_gib,
                ) = benchmark_one_config(
                    seq_len,
                    d_model,
                )

                print(
                    f"{d_model:>8} "
                    f"{seq_len:>10} "
                    f"{forward_ms:>15.3f} "
                    f"{backward_ms:>15.3f} "
                    f"{memory_gib:>15.3f} "
                    f"{'OK':>10}"
                )

            except torch.OutOfMemoryError:

                # ------------------------------------------------
                # OOM 也是题目要求报告的实验结果。
                # ------------------------------------------------

                print(
                    f"{d_model:>8} "
                    f"{seq_len:>10} "
                    f"{'OOM':>15} "
                    f"{'OOM':>15} "
                    f"{'OOM':>15} "
                    f"{'OOM':>10}"
                )

                clear_cuda()


if __name__ == "__main__":
    main()