import math

import triton
import triton.language as tl
import torch

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_i = tl.load(Q_block_ptr)
    O_i = tl.zeros(
        (Q_TILE_SIZE, D),
        dtype=tl.float32,
    )
    l_i = tl.zeros(
        (Q_TILE_SIZE, ),
        dtype=tl.float32,
    )
    m_i = tl.full(
        (Q_TILE_SIZE, ),
        value=-float("inf"),
        dtype=tl.float32,
    )

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_i_j = tl.dot(
            Q_i,
            tl.trans(K_j),
        ) * scale

        tile_max = tl.max(S_i_j, axis=1)
        m_i_new = tl.maximum(m_i, tile_max)

        P_tilde = tl.exp(S_i_j - m_i_new[:, None])
        correction = tl.exp(m_i - m_i_new)
        l_i_new = correction * l_i + tl.sum(P_tilde, axis=1)

        O_i = O_i * correction[:, None]
        P_tilde_for_dot = P_tilde.to(V_j.dtype)
        O_i = tl.dot(P_tilde_for_dot, V_j, acc=O_i)

        m_i = m_i_new
        l_i = l_i_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    # 写回前转换成对应 output buffer 的 dtype
    O_i = O_i.to(O_block_ptr.type.element_ty)
    L_i = L_i.to(L_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_i)
    tl.store(L_block_ptr, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
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
        )

        scale = 1.0 / math.sqrt(D)
        grid = (
            triton.cdiv(N_QUERIES, Q_TILE_SIZE),
            batch_size,
        )

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,

            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1), 

            N_QUERIES,N_KEYS,

            scale,

            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(
            L, Q, K, V, O
        )

        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

