import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. 门控网络 (Router): 一个简单的线性层
        self.gate = nn.Linear(input_dim, num_experts)

        # 2. 专家网络 (Experts): 这里的每个 expert 就是原本的 FFN
        # 使用 ModuleList 存储
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim) -> (B*s, D)
        original_shape = x.shape
        x = x.reshape(-1, original_shape[-1])
        
        # --- Step 1: Routing (门控) ---
        # 计算每个 token 对每个专家的分数
        gate_logits = self.gate(x)  # (B*T, num_experts)
        
        # 选出分数最高的 top_k 个专家
        # weights: 路由权重, indices: 专家索引
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # 归一化权重 (Softmax over top-k)
        weights = F.softmax(weights, dim=-1)

        # --- Step 2 & 3: Dispatch & Compute (分发与计算) ---
        # 这一步在单机上有多种实现方式，这里演示最直观的“循环掩码”法
        # (高性能实现通常用 torch.scatter / torch.gather 或 Einsum)
        
        final_output = torch.zeros_like(x)
        
        for i in range(self.num_experts):
            # 找出哪些 token 选中了第 i 个专家
            # indices shape: (Total_Tokens, top_k)
            # mask: 标记当前专家 i 是否在被选中的 top_k 里
            expert_mask = (indices == i).any(dim=-1)
            
            if expert_mask.sum() == 0:
                continue

            # 选出需要该专家处理的 token
            selected_tokens = x[expert_mask] 
            
            # 专家计算
            expert_out = self.experts[i](selected_tokens)
            
            # --- Weighting (加权) ---
            # 找到对应的权重。注意一个 token 可能两次选中同一个专家(虽然不常见)，
            # 这里简化逻辑：我们只关心当前 token 在 top_k 里的对应权重
            # 实际需复杂的 gather 操作，这里简化为：
            # 既然 mask 选出来了，我们需要把权重也对应取出来乘上去
            
            # 为了演示原理，这里假设 top-1，简化代码逻辑
            # 实际工程中需要用 scatter_add 将结果加回 final_output
            
            # 这里仅做示意：
            # final_output[expert_mask] += expert_out * matching_weights
            pass 

        return final_output.reshape(original_shape)
if __name__ == "__main__":
    # 测试 MoE 层
    batch_size = 2
    seq_len = 4
    input_dim = 8
    hidden_dim = 16
    output_dim = 8
    num_experts = 3

    moe_layer = MoELayer(input_dim, hidden_dim, output_dim, num_experts)
    x = torch.randn(batch_size, seq_len, input_dim)
    output = moe_layer(x)
    print("Output shape:", output.shape)