import time
import torch
import argparse
import statistics

# NVTX 标记，用于 Nsight Systems 可视化
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
# =====================================================
# 模型配置
# =====================================================

MODEL_CONFIG = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    }
}
# =====================================================
# 创建模型
# =====================================================

def create_model(model_size):
    config = MODEL_CONFIG[model_size]
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],

    )
    return model
# =====================================================
# 创建输入数据
# =====================================================

def create_batch(
    batch_size=4,
    vocab_size=10000,
    context_length=512
):

    tokens = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, context_length),
        device="cuda"
    )
    return tokens

# =====================================================
# Benchmark
# =====================================================

def benchmark(
    model,
    input_ids,
    precision="fp32",
    warmup_steps=5,
    measure_steps=10,

):
    times = []
    # -------------------------------------------------
    # 根据 precision 选择 autocast
    # -------------------------------------------------
    if precision == "fp16":
        dtype = torch.float16
    elif precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = None

 # -------------------------------------------------
    # 单步运行
    # -------------------------------------------------

    def run_step():

        with nvtx.range("forward"):
        # FP32
            if dtype is None:
                with nvtx.range("FP32_forward"):
                    logits = model(input_ids)

        # FP16 / BF16
            else:
                with nvtx.range(f"{precision}_forward"):
                    with torch.autocast(
                        device_type="cuda",
                        dtype=dtype

                    ):
                        logits = model(input_ids)

        # 简单 loss
        with nvtx.range("loss"):
            loss = logits.mean()
        return loss
    
    for _ in range(warmup_steps):
        loss = run_step()
        torch.cuda.synchronize()

    with nvtx.range("benchmark"):
        for _ in range(measure_steps):
            torch.cuda.synchronize()
            start = time.time()
            loss = run_step()
            torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)

    return times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(

        "--precision",
        type=str,
        default="fp32",
        choices=[
            "fp32",
            "fp16",
            "bf16"
        ]

    )

    args = parser.parse_args()
    # 创建模型

    model = create_model("small")
    model = model.cuda()
    model.eval()

    input_ids = create_batch()
    # 清空显存统计
    torch.cuda.reset_peak_memory_stats()
    # 开始 benchmark

    times = benchmark(
        model,
        input_ids,
        precision=args.precision
    )

    # 获取最大显存
    memory = (
        torch.cuda.max_memory_allocated()
        /1024**3
    )


    print("======================")

    print(
        "Precision:",
        args.precision
    )
    print(
        "Average time:",
        statistics.mean(times),
        "seconds"
    )
    print(
        "Std:",
        statistics.stdev(times)
    )
    print(
        "Peak memory:",
        memory,
        "GB"
    )
    print("======================")

if __name__ == "__main__":
    main()