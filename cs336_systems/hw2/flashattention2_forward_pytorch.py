import torch
import math

class FlashAttentionPyTorch(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # Q: [batch_size, N_QUERIES, D]
        # K: [batch_size, N_KEYS, D]
        # V: [batch_size, N_KEYS, D]
        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[1]
        N_KEYS = K.shape[1]
        D = Q.shape[2]

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        O = torch.empty_like(Q)

        L = torch.empty(
            (batch_size, N_QUERIES),
            device=Q.device,
            dtype=Q.dtype,
        )

        for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
            q_end = q_start + Q_TILE_SIZE
            # 当前 query tile:
            # Q_i: [B, B_q, D]
            Q_i = Q[:, q_start:q_end, :]

            # O_i^(0): [B, B_q, D]
            O_i = torch.zeros(
                (batch_size, Q_TILE_SIZE, D),
                device=Q.device,
                dtype=Q.dtype,
            )

            # l_i^(0): [B, B_q]
            l_i = torch.zeros(
                (batch_size, Q_TILE_SIZE),
                device=Q.device,
                dtype=Q.dtype,
            )

            # m_i^(0): [B, B_q]
            m_i = torch.full(
                (batch_size, Q_TILE_SIZE),
                float("-inf"),
                device=Q.device,
                dtype=Q.dtype,
            )

            for k_start in range(0, N_KEYS, K_TILE_SIZE):
                k_end = k_start + K_TILE_SIZE

                K_j = K[:, k_start:k_end, :]
                V_j = V[:, k_start:k_end, :]

                S_i_j = torch.matmul(
                    Q_i, K_j.transpose(-2, -1),
                ) / math.sqrt(D)

                # --------------------------------------------------
                # Algorithm line 10:
                # 更新当前每个 query row 的 running maximum
                # --------------------------------------------------
                tile_max = S_i_j.max(dim=-1).values

                m_i_new = torch.maximum(
                    m_i,
                    tile_max,
                )

                # --------------------------------------------------
                # Algorithm line 11:
                # 当前 tile 的未归一化 softmax numerator
                # --------------------------------------------------
                P_tilde = torch.exp(
                    S_i_j - m_i_new.unsqueeze(-1)
                )

                correction = torch.exp(m_i - m_i_new)

                l_i_new = (
                    correction * l_i + P_tilde.sum(dim=-1)
                )

                O_i_new = (
                    correction.unsqueeze(-1) * O_i 
                    + torch.matmul(P_tilde, V_j)
                )

                m_i = m_i_new
                l_i = l_i_new
                O_i = O_i_new

            O_i = O_i / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i)
            O[:, q_start:q_end, :] = O_i
            L[:, q_start:q_end] = L_i

        # 保存给之后的 backward
        ctx.save_for_backward(L, Q, K, V, O)

        # (a) 要求 forward 返回 O
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


def attention_reference(Q, K, V):
    D = Q.shape[-1]

    S = torch.matmul(Q, K.transpose(-2, -1), ) / math.sqrt(D)
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    L = torch.logsumexp(S, dim=-1)

    return O, L


B = 2
N = 32
D = 16

Q = torch.randn(B, N, D, device="cuda")
K = torch.randn(B, N, D, device="cuda")
V = torch.randn(B, N, D, device="cuda")

# 我们刚刚实现的 tiled FlashAttention
O_flash = FlashAttentionPyTorch.apply(Q, K, V, False)

# 普通 attention reference
O_ref, L_ref = attention_reference(Q, K, V)

print("O close:",
      torch.allclose(O_flash, O_ref, atol=1e-5, rtol=1e-5))

print("max O error:",
      (O_flash - O_ref).abs().max().item())