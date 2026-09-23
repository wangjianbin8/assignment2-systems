import timeit
import torch
from cs336_basics.model import BasicsTransformerLM
import statistics
import argparse
import torch.cuda.nvtx as nvtx

MODEL_CONFIGS = {


    "small": {

        "d_model":768,
        "d_ff":3072,
        "num_layers":12,
        "num_heads":12,
    },


    "medium": {

        "d_model":1024,
        "d_ff":4096,
        "num_layers":12,
        "num_heads":16,
    },


    "large": {

        "d_model":1280,
        "d_ff":5120,
        "num_layers":36,
        "num_heads":20,
    },


    "xl": {

        "d_model":2560,
        "d_ff":10240,
        "num_layers":32,
        "num_heads":32,
    },


    "10B": {

        "d_model":4608,
        "d_ff":12288,
        "num_layers":50,
        "num_heads":36,
    },

}

def parse_args():
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(
        description="Benchmark Transformer model"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=[
            "small",
            "medium",
            "large",
            "xl",
            "10B",
        ],
        help="Which model size to benchmark",
    )    

    parser.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=[
            "forward",
            "forward_backward",
            "train",
        ],
        help="What part to benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    return args


def create_model(model_size):
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=10000,
        context_length=512,
        # 从配置表读取
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    return model

def create_batch(
    batch_size = 4,
    vocab_size = 10000,
    context_length = 512
):
    tokens = torch.randint(
        low = 0, high=vocab_size, size=(batch_size, context_length), device="cuda"
    )
    return tokens

def benchmark(
    model,
    input_ids,
    optimizer=None,
    mode="forward",
    warmup_steps=5,
    measure_steps=10,
):
    times = []
    def run_step():


        # ==========================
        # Forward
        # ==========================

        with nvtx.range("forward"):

            logits = model(input_ids)



        # 如果只测 forward
        # 直接返回

        if mode == "forward":
            return



        # ==========================
        # Backward
        # ==========================

        loss = logits.mean()


        with nvtx.range("backward"):

            loss.backward()



        # ==========================
        # Optimizer
        # ==========================

        if mode == "train":

            with nvtx.range("optimizer"):

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

    for _ in range(warmup_steps):

        run_step()

        torch.cuda.synchronize()


    with nvtx.range("benchmark"):

        for i in range(measure_steps):

            start = timeit.default_timer()


            run_step()


            torch.cuda.synchronize()


            end = timeit.default_timer()


            times.append(
                end-start
            )
    return times


def main():
    args = parse_args()

    model = create_model(args.model_size)
    model = model.cuda()
    input_ids = create_batch()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )

    times = benchmark(
        model=model,
        input_ids=input_ids,
        optimizer=optimizer,
        mode=args.mode,
        warmup_steps=args.warmup,
        measure_steps=args.steps,
    )
    print(
        "mean:",
        statistics.mean(times)
    )


    print(
        "std:",
        statistics.stdev(times)
    )


if __name__ == "__main__":
    main()
