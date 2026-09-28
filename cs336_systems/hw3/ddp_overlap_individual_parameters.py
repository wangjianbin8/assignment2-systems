
'''
和 naive DDP 的区别
之前 naive：
forward

↓

backward

↓

等待所有 gradient

↓

all_reduce(param1.grad)

all_reduce(param2.grad)

...

↓

optimizer.step()

现在：
forward

↓

backward

↓

param1.grad ready
        |
        |
        hook
        |
        |
        async all_reduce


param2.grad ready
        |
        |
        hook
        |
        |
        async all_reduce


↓

wait all communication

↓

optimizer.step()
'''
import torch
import torch.distributed as dist


class DDP(torch.nn.Module):
    """
    Distributed Data Parallel with overlapping
    individual parameter gradient communication.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self.module = module

        # 保存异步通信 handle
        self._handles = []

        # 1. 初始化参数同步
        self._broadcast_parameters()

        # 2. 注册 gradient hook
        self._register_gradient_hooks()


    def forward(self, *inputs, **kwargs):
        """
        Forward pass through wrapped module.
        """

        return self.module(
            *inputs,
            **kwargs
        )


    def _broadcast_parameters(self):
        """
        Broadcast parameters from rank 0
        so every worker starts with identical weights.
        """

        for param in self.module.parameters():

            dist.broadcast(
                param.data,
                src=0,
            )


    def _register_gradient_hooks(self):
        """
        Register hooks on every parameter.

        When gradient accumulation finishes,
        asynchronously all-reduce that gradient.
        """

        for param in self.module.parameters():

            # 没有 requires_grad 的参数不用同步
            if not param.requires_grad:
                continue


            param.register_post_accumulate_grad_hook(
                self._make_hook(param)
            )


    def _make_hook(self, param):
        """
        Create hook function for one parameter.
        """

        def hook(param_tensor):

            # 理论上这里 param.grad 已经 ready
            if param.grad is None:
                return


            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )


            self._handles.append(
                (handle, param)
            )


        return hook



    def finish_gradient_synchronization(self):
        """
        Wait for all async all-reduce operations
        to finish.

        Then average gradients.
        """


        # 等待通信完成
        for handle, _ in self._handles:

            handle.wait()



        # all_reduce 是 sum
        # 所以需要除 world size
        world_size = dist.get_world_size()


        for _, param in self._handles:

            if param.grad is not None:

                param.grad /= world_size



        # 清空，防止下一 iteration 重复等待
        self._handles.clear()