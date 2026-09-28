import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    torch.cuda.set_device(rank)



def cleanup():

    dist.destroy_process_group()



def benchmark(
    rank,
    world_size,
    tensor_MB,
):

    setup(
        rank,
        world_size,
    )


    num_elements = (
        tensor_MB
        * 1024
        * 1024
        // 4
    )


    x = torch.randn(
        num_elements,
        device=f"cuda:{rank}",
        dtype=torch.float32,
    )


    # -------------------------
    # warmup
    # -------------------------
    for _ in range(5):

        dist.all_reduce(
            x
        )


    torch.cuda.synchronize()


    # -------------------------
    # benchmark
    # -------------------------
    start = time.time()


    dist.all_reduce(
        x
    )


    torch.cuda.synchronize()


    end = time.time()


    elapsed = (
        end-start
    ) * 1000



    # -------------------------
    # collect max time
    # -------------------------
    t = torch.tensor(
        [elapsed],
        device=x.device,
    )


    dist.reduce(
        t,
        dst=0,
        op=dist.ReduceOp.MAX,
    )


    if rank == 0:

        print(
            f"GPUs={world_size}, "
            f"Size={tensor_MB}MB, "
            f"Time={t.item():.3f}ms"
        )


    cleanup()



if __name__ == "__main__":


    sizes = [
        1,
        10,
        100,
        1024,
    ]


    gpu_counts = [
        2,
        4,
        6,
    ]


    for world_size in gpu_counts:

        for size in sizes:

            mp.spawn(
                benchmark,
                args=(
                    world_size,
                    size,
                ),
                nprocs=world_size,
                join=True,
            )