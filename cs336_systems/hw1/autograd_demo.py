import torch
from cs336_systems.hw1.model import RMSNorm

x = torch.randn(2, 4, 8, requires_grad=True)
norm = RMSNorm(hidden_size=8)

def pack_hook(tensor):
    print("保存：", tensor.shape)
    return tensor


def unpack_hook(tensor):
    print("读取：", tensor.shape)
    return tensor

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = norm(x)
    loss = y.sum()

    print("forward 完成")

    loss.backward()

print("x.grad =", x.grad)