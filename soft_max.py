import triton   
import triton.language as tl

@triton.jit
def soft_max_kernel(
    x_ptr,
    output_ptr,
    N,   # 输入元素的数量
    BLOCK_SIZE: tl.constexpr,
):
    pid  = tl.program_id(0)
    block_start = pid * BLOCK_SIZE  
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N  
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_x = tl.max(x, axis=0)
    x_exp = tl.exp(x - max_x)
    sum_x_exp = tl.sum(x_exp, axis=0)
    softmax_x = x_exp / sum_x_exp
    tl.store(output_ptr + offsets, softmax_x, mask=mask)
def soft_max_tensors(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    output = torch.empty_like(x)
    soft_max_kernel[grid](x_ptr=x, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE)
    return output