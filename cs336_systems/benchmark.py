import time
import sys

import torch
import cs336_basics.model


MODEL_CONFIGS = {
    "tiny": {
        "d_model": 640,
        "d_ff": 2560,
        "num_layers": 8,
        "num_heads": 10,
    },

    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
}

def print_progress(current, total, stage):
    progress = current / total
    bar_len = 20
    filled = int(bar_len * progress)

    bar = "█" * filled + "-" * (bar_len - filled)

    sys.stdout.write(
        f"\r[{bar}] {progress * 100:5.1f}% | {stage} {current}/{total}"
    )
    sys.stdout.flush()


def benchmark_model(
    model_size="tiny",
    warmup_steps=2,
    measure_steps=5,
    use_bf16=False,
):

    config = MODEL_CONFIGS[model_size]

    batch_size = 4
    context_length = 512
    vocab_size = 10000


    # ----------------------
    # create model
    # ----------------------

    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        **config,
    ).cuda()

    model.train()


    optimizer = torch.optim.AdamW(
        model.parameters()
    )


    x = torch.randint(
        0,
        vocab_size,
        (batch_size, context_length),
        device="cuda",
    )


    precision = "BF16" if use_bf16 else "FP32"

    print(
        f"\nTesting {model_size.upper()} | {precision}"
    )


    # ----------------------
    # warmup
    # ----------------------

    for i in range(1, warmup_steps + 1):

        print_progress(
            i,
            warmup_steps,
            "Warmup"
        )


        optimizer.zero_grad()


        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):

            output = model(x)
            loss = output.sum()


        loss.backward()

        optimizer.step()


    torch.cuda.synchronize()

    print(" ✔")


    # ----------------------
    # measure
    # ----------------------

    forward_times = []
    backward_times = []
    optimizer_times = []


    for i in range(1, measure_steps + 1):

        print_progress(
            i,
            measure_steps,
            "Measure"
        )


        optimizer.zero_grad()


        # ---- forward ----

        torch.cuda.synchronize()

        start = time.perf_counter()


        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):

            output = model(x)
            loss = output.sum()


        torch.cuda.synchronize()

        forward_times.append(
            time.perf_counter() - start
        )


        # ---- backward ----

        torch.cuda.synchronize()

        start = time.perf_counter()


        loss.backward()


        torch.cuda.synchronize()

        backward_times.append(
            time.perf_counter() - start
        )


        # ---- optimizer step ----

        torch.cuda.synchronize()

        start = time.perf_counter()


        optimizer.step()


        torch.cuda.synchronize()

        optimizer_times.append(
            time.perf_counter() - start
        )


    print(" ✔")


    # ----------------------
    # result
    # ----------------------

    forward_avg = sum(forward_times) / measure_steps
    backward_avg = sum(backward_times) / measure_steps
    optimizer_avg = sum(optimizer_times) / measure_steps


    print(
        f"""
Results:

Forward:
    {forward_avg:.6f}s

Backward:
    {backward_avg:.6f}s

Optimizer:
    {optimizer_avg:.6f}s

Total:
    {forward_avg + backward_avg + optimizer_avg:.6f}s
"""
    )


if __name__ == "__main__":

    print(
        "=== FP32 Benchmark ==="
    )

    benchmark_model(
        "tiny",
        use_bf16=False
    )


    print(
        "\n=== BF16 Benchmark ==="
    )

    benchmark_model(
        "tiny",
        use_bf16=True
    )