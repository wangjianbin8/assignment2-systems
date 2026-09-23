import argparse
from contextlib import nullcontext

import torch

from cs336_basics.model import BasicsTransformerLM


# ============================================================
# XL 模型配置
# ============================================================
#
# 严格使用讲义 Table 1 的 xl 配置：
#
# d_model    = 2560
# d_ff       = 10240
# num_layers = 32
# num_heads  = 32
#
# vocab_size 和 batch_size 按讲义默认：
#
# vocab_size = 10000
# batch_size = 4
#
# context_length 通过命令行传入：
#
# 128 或 2048
# ============================================================

XL_CONFIG = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 8,
    "num_heads": 8,
}


def parse_args():
    """
    读取命令行参数。
    """

    parser = argparse.ArgumentParser(
        description="CS336 memory profiling"
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
    )

    # --------------------------------------------------------
    # context length
    #
    # 讲义要求后面比较 128 和 2048。
    # --------------------------------------------------------
    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        choices=[128, 2048],
    )

    # --------------------------------------------------------
    # forward:
    #     只做 inference / forward
    #
    # train:
    #     forward + backward + optimizer step
    # --------------------------------------------------------
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["forward", "train"],
    )

    # --------------------------------------------------------
    # 先预留 precision 参数。
    #
    # Part (a) 先使用 fp32。
    # Part (c) 会重新使用 bf16 mixed precision。
    # --------------------------------------------------------
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
    )

    # 输出 memory snapshot 文件名
    parser.add_argument(
        "--output",
        type=str,
        default="memory_snapshot.pickle",
    )

    return parser.parse_args()


def create_model(context_length):
    """
    创建讲义规定的 XL Transformer。
    """

    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=context_length,
        d_model=XL_CONFIG["d_model"],
        num_layers=XL_CONFIG["num_layers"],
        num_heads=XL_CONFIG["num_heads"],
        d_ff=XL_CONFIG["d_ff"],
    )

    return model.cuda()


def create_batch(context_length):
    """
    创建随机 token batch。

    shape:
        [batch_size, context_length]

    讲义默认 batch_size = 4。
    """

    input_ids = torch.randint(
        low=0,
        high=10000,
        size=(4, context_length),
        device="cuda",
    )

    return input_ids


def get_autocast_context(precision):
    """
    根据 precision 返回 context manager。

    FP32:
        什么都不做，因此使用 nullcontext()

    BF16:
        使用 torch.autocast
    """

    if precision == "bf16":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return nullcontext()


def run_forward(model, input_ids, precision):
    """
    只执行 forward。

    这里用 torch.no_grad()，
    因为这一模式对应 inference-only。
    """

    with torch.no_grad():

        with get_autocast_context(precision):

            logits = model(input_ids)

    return logits


def run_training_step(
    model,
    input_ids,
    optimizer,
    precision,
):
    """
    完整训练步骤：

        forward
        ↓
        loss
        ↓
        backward
        ↓
        optimizer.step()
        ↓
        zero_grad()
    """

    # ----------------------------
    # Forward
    # ----------------------------
    with get_autocast_context(precision):

        logits = model(input_ids)

        # benchmark / profiling 只关心计算和显存，
        # 所以使用一个简单的标量 loss。
        loss = logits.float().mean()

    # ----------------------------
    # Backward
    # ----------------------------
    loss.backward()

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)


def main():
    args = parse_args()

    print(f"mode           = {args.mode}")
    print(f"context_length = {args.context_length}")
    print(f"precision      = {args.precision}")

    # ========================================================
    # 1. 创建模型和输入
    # ========================================================

    model = create_model(args.context_length)

    input_ids = create_batch(args.context_length)

    optimizer = None

    if args.mode == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
        )

    # ========================================================
    # 2. Warm-up
    #
    # 重要：
    # 按讲义要求，memory history 应该在 warm-up 后开始。
    # 否则初始化产生的显存分配会污染我们的 timeline。
    # ========================================================

    print("Running warm-up...")
    for _ in range(args.warmup_steps):
        if args.mode == "forward":

            run_forward(
                model,
                input_ids,
                args.precision,
            )

        else:

            run_training_step(
                model,
                input_ids,
                optimizer,
                args.precision,
            )

    # 等 GPU 真正执行完成
    torch.cuda.synchronize()

    print("Warm-up finished.")


    # ========================================================
    # 重置 peak memory 统计
    #
    # 这样后面得到的 peak，
    # 就主要对应我们真正要 profile 的这一次运行。
    # ========================================================
    torch.cuda.reset_peak_memory_stats()

    # ========================================================
    # 3. 开始记录显存历史
    # ========================================================

    torch.cuda.memory._record_memory_history(
        max_entries=1_000_000
    )

    print("Memory recording started.")

    # ========================================================
    # 4. 真正需要 profile 的一次运行
    # ========================================================

    if args.mode == "forward":

        run_forward(
            model,
            input_ids,
            args.precision,
        )

    else:

        run_training_step(
            model,
            input_ids,
            optimizer,
            args.precision,
        )

    torch.cuda.synchronize()

    # ========================================================
    # 获取这次正式运行期间的 peak memory
    # ========================================================
    peak_memory_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    print(
        f"Peak memory: {peak_memory_gib:.3f} GiB"
    )

    # ========================================================
    # 5. 导出 snapshot
    # ========================================================

    torch.cuda.memory._dump_snapshot(
        args.output
    )

    # ========================================================
    # 6. 停止记录
    # ========================================================

    torch.cuda.memory._record_memory_history(
        enabled=None
    )

    print(
        f"Saved memory snapshot to: {args.output}"
    )


if __name__ == "__main__":
    main()