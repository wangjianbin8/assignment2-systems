import math
import torch

def flash_backward_pytorch(Q, K, V, O, dO, L, is_causal=False):
    """
    FlashAttention-2 backward computation
    following Equations 13--19.
    """

    d = Q.shape[-1]

    scale = 1.0 / math.sqrt(d)

    D_vec = torch.sum(O * dO, dim=-1)

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if is_causal:
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]

        q_indices = torch.arange(N_QUERIES, device=Q.device)
        k_indices = torch.arange(N_KEYS, device=Q.device)

        causal_mask = (k_indices[None, :] > q_indices[:, None])
        # masked_fill 会保持 S 自己的 dtype
        S = S.masked_fill(
            causal_mask,
            -1e6,
        )

    # L 从 Triton forward 中可能是 float32，
    # 转成与 S 相同的 dtype，避免后续 matmul dtype 不一致。
    P = torch.exp(S - L.to(S.dtype).unsqueeze(-1))

    dV = torch.matmul(P.transpose(-2, -1), dO)

    dP = torch.matmul(dO, V.transpose(-2, -1))

    dS = P * (dP - D_vec.unsqueeze(-1))

    dQ = torch.matmul(dS, K) * scale

    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV

flash_backward_compiled = torch.compile(flash_backward_pytorch)