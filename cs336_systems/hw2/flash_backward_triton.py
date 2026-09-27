import triton
import triton.language as tl

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
        P_i_j = tl.exp(S_i_j - L_i[:, None])
        dP_i_j = tl.dot(dO_i, tl.trans(V_j))
        dS_i_j = P_i_j * (dP_i_j - D_i[:, None])

        dS_for_dot = dS_i_j.to(K_j.dtype)
        dQ_i += (tl.dot(dS_for_dot, K_j) * scale)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    dQ_i = dQ_i.to(dQ_block_ptr.type.element_ty)
    tl.store(dQ_block_ptr, dQ_i)


