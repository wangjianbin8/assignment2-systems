import torch

class NaiveDDP(torch.nn.Module):

    def broadcast_parameters(self):
        for param in self.module.parameters():
            torch.dist.broadcast(
                param.data,
                src=0,
            )

    def __init__(self, module):
        super().__init__()
        self.module = module

        self.broadcast_parameters()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    def finish_gradient_synchronization(self):

        world_size = torch.dist.get_world_size()

        for p in self.module.parameters():

            if p.grad is not None:

                torch.dist.all_reduce(
                    p.grad,
                    op=torch.dist.ReduceOp.SUM,
                )

                p.grad /= world_size