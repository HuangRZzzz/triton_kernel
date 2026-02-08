import triton 
import triton.language as tl



@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    N,   # 输入元素的数量
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    result = x + y
    tl.store(output_ptr + offsets,result ,mask = mask)

def add_tensors(x, y):
    assert x.shape == y.shape
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ( (N + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    output = torch.empty_like(x)
    add_kernel[grid](x_ptr=x, y_ptr=y, output_ptr=output, N=N, BLOCK_SIZE=BLOCK_SIZE
                     )



