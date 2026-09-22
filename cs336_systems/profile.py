import torch
from torch.profiler import profile, ProfilerActivity

import cs336_basics.model


model = cs336_basics.model.BasicsTransformerLM(
    vocab_size=10000,
    context_length=512,
    d_model=640,
    d_ff=2560,
    num_layers=8,
    num_heads=10,
).cuda()


optimizer = torch.optim.AdamW(
    model.parameters()
)


x = torch.randint(
    0,
    10000,
    (4,512),
    device="cuda"
)


model.train()


# warmup
for _ in range(3):
    optimizer.zero_grad()

    output = model(x)

    loss = output.sum()

    loss.backward()

    optimizer.step()


torch.cuda.synchronize()


with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:


    optimizer.zero_grad()

    output = model(x)

    loss = output.sum()

    loss.backward()

    optimizer.step()



print(
    prof.key_averages()
    .table(
        sort_by="cuda_time_total",
        row_limit=20
    )
)