import torch
import triton
import triton.language as tl
import math

@triton.jit
def _flash_attention_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    """
    FlashAttention Forward Kernel
    Q, K, V: [Batch, Head, SeqLen, Dim]
    """
    # 1. 确定当前程序的 ID
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    
    # 确定 Batch 和 Head 的索引
    off_z = off_hz // H
    off_h = off_hz % H

    # 2. 计算 Q 的指针偏移
    # Q 的形状通常是 [Z, H, M, D]
    q_offset = (off_z * stride_qz + off_h * stride_qh)
    k_offset = (off_z * stride_kz + off_h * stride_kh)
    v_offset = (off_z * stride_vz + off_h * stride_vh)
    o_offset = (off_z * stride_oz + off_h * stride_oh)

    # Q 矩阵的 block 指针
    # range_m: [0, 1, ..., BLOCK_M-1]
    # range_d: [0, 1, ..., BLOCK_DMODEL-1]
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # 指向 Q 的具体数据块
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)

    # 3. 加载 Q 到 SRAM
    # mask 用于处理序列长度不能被 block 整除的情况 (简化起见，这里假设能整除或由外部padding)
    q = tl.load(q_ptrs)

    # 4. 初始化累加器
    # m_i: 运行中的最大值 (初始化为负无穷)
    # l_i: 运行中的指数和 (初始化为 1.0)
    # acc: 输出累加器 (初始化为 0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # 5. 缩放系数 (通常是 1 / sqrt(d_model))
    qk_scale = sm_scale

    # 6. 循环遍历 K 和 V 的 block (外层循环是 Q 的分块，内层循环遍历所有的 K, V)
    # 这里的循环是沿着 N 维度 (SeqLen of K/V)
    # lo 和 hi 定义了 causal masking (如果是 causal attention) 或者全长 (如果是普通 attention)
    # 这里我们实现标准的非 causal attention，遍历所有 K, V
    
    offs_n = tl.arange(0, BLOCK_N)
    # K, V 的指针基准
    k_ptrs_base = K + k_offset + (offs_d[None, :] * stride_kk)
    v_ptrs_base = V + v_offset + (offs_d[None, :] * stride_vk) # V 通常转置存储或读取逻辑不同，这里假设 Row Major

    # 迭代 steps
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        
        # --- Load K ---
        # K shape: [BLOCK_N, BLOCK_DMODEL] (我们需要转置来做点积，或者直接按列加载)
        # 这里的 k_ptrs 指向 K 的 block
        # 注意: 为了优化内存访问，K 在内存中通常是 [N, D]，我们需要 K^T [D, N]
        # 但 tl.dot 支持操作数，我们按正常加载，tl.dot 会处理
        cols_n = start_n + offs_n
        k_ptrs = k_ptrs_base + (cols_n[:, None] * stride_kn)
        k = tl.load(k_ptrs)
        
        # --- Load V ---
        v_ptrs = v_ptrs_base + (cols_n[:, None] * stride_vn)
        v = tl.load(v_ptrs)

        # --- Compute QK^T ---
        # q: [BLOCK_M, BLOCK_D], k: [BLOCK_N, BLOCK_D]
        # qk: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k))
        qk *= qk_scale

        # --- Online Softmax Logic ---
        
        # 1. 计算当前 block 的最大值
        m_ij = tl.max(qk, 1) # 沿着 N 维度找最大值
        
        # 2. 更新全局最大值 m_new
        m_new = tl.maximum(m_i, m_ij)
        
        # 3. 计算缩放因子
        # alpha: 用于缩放旧的 acc 和 l (因为最大值变大了)
        # beta:  用于缩放当前的 exp(qk) (因为减去了新的最大值)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(qk - m_new[:, None])
        
        # 4. 更新累加器 acc
        # acc_new = acc_old * alpha + beta * V
        acc = acc * alpha[:, None] 
        acc += tl.dot(beta.to(tl.float16), v) # 将 beta 转回 float16 以加速 dot
        
        # 5. 更新分母 l
        l_i = l_i * alpha + tl.sum(beta, 1)
        
        # 6. 更新运行中的最大值
        m_i = m_new

    # 7. 最终归一化
    # acc / l_i
    acc = acc / l_i[:, None]

    # 8. 存储结果
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_on)
    tl.store(o_ptrs, acc.to(tl.float16))

# Python 包装器
def flash_attention(q, k, v, sm_scale):
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_DMODEL = q.shape[-1]
    
    # 形状推断
    batch, head, seq_len, d_head = q.shape
    
    # 输出张量
    o = torch.empty_like(q)

    
    # Grid 维度: (M轴分块数, Batch * Head)
    grid = (triton.cdiv(seq_len, BLOCK_M), batch * head)
    
    # 启动 Kernel
    _flash_attention_fwd_kernel[grid](
        q, k, v, sm_scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        batch, head, seq_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )
    return o

# --- 验证代码 ---
def test_flash_attention():
    torch.manual_seed(0)
    Z, H, N_CTX, D_HEAD = 1, 4, 1024, 64
    dtype = torch.float16
    device = "cuda"

    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=False)
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # 1. Triton 结果
    tri_out = flash_attention(q, k, v, sm_scale)

    # 2. PyTorch 原生结果 (Reference)
    # 维度转换: [Z, H, N, D] -> [Z, H, N, D]
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=sm_scale)

    # 3. 比较
    print(f"Triton Out Sample: {tri_out[0,0,0,:5]}")
    print(f"Torch  Out Sample: {ref_out[0,0,0,:5]}")
    
    diff = torch.max(torch.abs(tri_out - ref_out))
    print(f"Max Difference: {diff.item()}")
    
    if diff < 1e-2: # float16 精度误差通常在这个范围
        print("✅ Test Passed!")
    else:
        print("❌ Test Failed!")

if __name__ == "__main__":
    test_flash_attention()