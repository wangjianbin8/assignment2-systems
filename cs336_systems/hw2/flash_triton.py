import math

import triton
import triton.language as tl
import torch
# from cs336_systems.hw2.flash_backward_pytorch import flash_backward_compiled

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

        # ============================================================
        # Causal masking
        #
        # A query at position q can only attend to keys k <= q.
        # Therefore, positions satisfying k > q are masked.
        # ============================================================
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = k_indices[None, :] > q_indices[:, None]

            S_i_j += tl.where(causal_mask, -1e6, 0.0)

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

#先得到D
@triton.jit
def flash_bwd_preprocess_kernel(
    O_ptr,
    dO_ptr,
    D_ptr,

    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,

    N_QUERIES,

    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES, ),
        strides=(stride_dq, ),
        offsets=(query_tile_index * Q_TILE_SIZE),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    O_i = tl.load(O_block_ptr)
    dO_i = tl.load(dO_block_ptr)

    D_i = tl.sum(O_i * dO_i, axis=1)
    tl.store(D_block_ptr, D_i)

@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, 
    dO_ptr, 
    L_ptr,
    D_ptr, 
    dK_ptr, dV_ptr,

    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,

    N_QUERIES, N_KEYS,
    scale,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES, ),
        strides=(stride_Dq, ),
        offsets=(0, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    K_j = tl.load(K_block_ptr)
    V_j = tl.load(V_block_ptr)

    dK_j = tl.zeros(
        (K_TILE_SIZE, D_HEAD),
        dtype=tl.float32,
    )

    dV_j = tl.zeros(
        (K_TILE_SIZE, D_HEAD),
        dtype=tl.float32,
    )
    '''
    Q_i   → bf16
    K_j   → bf16
    V_j   → bf16
    dO_i  → bf16

    S     → 通常用更高精度计算/累积
    P     → 通常是 float32
    dS    → 通常是 float32
    '''

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        Q_i = tl.load(Q_block_ptr)
        dO_i = tl.load(dO_block_ptr)
        L_i = tl.load(L_block_ptr)
        D_i = tl.load(D_block_ptr)

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_indices = (q_start + tl.arange(0, Q_TILE_SIZE))
            k_indices = (key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))
            causal_mask = (k_indices[None, :] > q_indices[:, None])
            S_i_j += tl.where(causal_mask, -1e6, 0.0)

        P_i_j = tl.exp(S_i_j - L_i[:, None])

        P_for_dot = P_i_j.to(dO_i.dtype)
        dV_j += tl.dot(tl.trans(P_for_dot), dO_i)
        '''
            P          FP32
            ↓ 为了 tl.dot
            P_for_dot  BF16
                    @
            dO         BF16
            ↓
            结果累积进 FP32 dV_j
        '''
        dP_i_j = tl.dot(dO_i, tl.trans(V_j))
        dS_i_j = P_i_j * (dP_i_j - D_i[:, None])

        dS_for_dot = dS_i_j.to(Q_i.dtype)
        dK_j += (tl.dot(tl.trans(dS_for_dot), Q_i)) * scale

        Q_block_ptr = Q_block_ptr.advance(
            (Q_TILE_SIZE, 0)
        )
        dO_block_ptr = dO_block_ptr.advance(
            (Q_TILE_SIZE, 0)
        )
        L_block_ptr = L_block_ptr.advance(
            (Q_TILE_SIZE,)
        )
        D_block_ptr = D_block_ptr.advance(
            (Q_TILE_SIZE,)
        )

    #最后 dK_j/dV_j 是 FP32 accumulator，但你的输出 dK/dV 很可能跟 K/V 一样是 bf16
    dK_j = dK_j.to(
        dK_block_ptr.type.element_ty
    )

    dV_j = dV_j.to(
        dV_block_ptr.type.element_ty
    )

    tl.store(dK_block_ptr, dK_j)
    tl.store(dV_block_ptr, dV_j)


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, 
    dO_ptr, 
    L_ptr,
    D_ptr, 

    dQ_ptr,

    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dqb, stride_dqq, stride_dqd,

    N_QUERIES, N_KEYS,
    scale,
    D_HEAD: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES, ),
        strides=(stride_Dq, ),
        offsets=(query_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0, )
    )

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_HEAD),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    # 从第 0 个 K/V tile 开始
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_HEAD),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D_HEAD),
        order=(1, 0)
    )

    Q_i = tl.load(Q_block_ptr)
    dO_i = tl.load(dO_block_ptr)
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(D_block_ptr)

    dQ_i = tl.zeros((Q_TILE_SIZE, D_HEAD), dtype=tl.float32)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_i_j = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_indices = k_start + tl.arange(0, K_TILE_SIZE)
            causal_mask = (k_indices[None, :] > q_indices[:, None])
            S_i_j += tl.where(causal_mask, -1e6, 0.0)

        P_i_j = tl.exp(S_i_j - L_i[:, None])
        dP_i_j = tl.dot(dO_i, tl.trans(V_j))
        dS_i_j = P_i_j * (dP_i_j - D_i[:, None])

        dS_for_dot = dS_i_j.to(K_j.dtype)
        dQ_i += (tl.dot(dS_for_dot, K_j) * scale)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    dQ_i = dQ_i.to(dQ_block_ptr.type.element_ty)
    tl.store(dQ_block_ptr, dQ_i)


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
    def backward(ctx, dO):
        # ==================================================
        # 1. 取回 forward 保存的数据
        # ==================================================
        L, Q, K, V, O = ctx.saved_tensors

        is_causal = ctx.is_causal
        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[1]
        N_KEYS = K.shape[1]
        D_HEAD = Q.shape[2]

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        scale = 1.0 / math.sqrt(D_HEAD)
        # ==================================================
        # 2. 分配最终 gradients
        #
        # dQ 和 Q shape 一样
        # dK 和 K shape 一样
        # dV 和 V shape 一样
        # ==================================================
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        # ==================================================
        # 3. 分配 D vector
        #
        # D = rowsum(O ∘ dO)
        #
        # shape: [batch_size, N_QUERIES]
        #
        # 用 float32 存，因为它是 backward 中的重要
        # reduction/intermediate quantity。
        # ==================================================
        D_vec = torch.empty(
            (batch_size, N_QUERIES),
            device=Q.device,
            dtype=torch.float32,
        )
        # ==================================================
        # Phase 1:
        # preprocess D
        #
        # 一个 program 负责一个 query tile
        # ==================================================
        grid_D = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)
        flash_bwd_preprocess_kernel[grid_D](
            O, dO, D_vec, 
            O.stride(0), O.stride(1), O.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            D_vec.stride(0), D_vec.stride(1),
            N_QUERIES, D_HEAD=D_HEAD, Q_TILE_SIZE=Q_TILE_SIZE,
        )
        # ==================================================
        # Phase 2:
        # compute dK and dV
        #
        # 一个 program 固定一个 K/V tile，
        # 然后遍历所有 Q tiles。
        # ==================================================
        grid_dkdv = (triton.cdiv(N_KEYS, K_TILE_SIZE,), batch_size,)
        flash_bwd_dkdv_kernel[grid_dkdv](
            Q, K, V,
            dO,
            L,
            D_vec,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_QUERIES, N_KEYS,
            scale,
            D_HEAD=D_HEAD,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )
        # ==================================================
        # Phase 3:
        # compute dQ
        #
        # 一个 program 固定一个 Q tile，
        # 然后遍历所有 K/V tiles。
        # ==================================================
        grid_dq = (
            triton.cdiv(
                N_QUERIES,
                Q_TILE_SIZE,
            ),
            batch_size,
        )

        flash_bwd_dq_kernel[grid_dq](
            # tensors
            Q,
            K,
            V,

            dO,
            L,
            D_vec,

            dQ,

            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),

            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),

            N_QUERIES,
            N_KEYS,
            scale,
            D_HEAD=D_HEAD,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        # ==================================================
        # forward(ctx, Q, K, V, is_causal)
        #
        # 所以 backward 必须返回 4 个位置：
        #
        # Q         -> dQ
        # K         -> dK
        # V         -> dV
        # is_causal -> None
        # ==================================================
        return dQ, dK, dV, None
