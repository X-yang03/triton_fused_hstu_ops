import torch
import os
import torch.nn.functional as F
from fused_hstu_v1 import hstu_fused_attention
from fused_hstu_v2 import hstu_fused_attention_v2

def silu(x):
    # 这里用常数近似 e = 2.718281828459045
    return x / (1 + 2.718281828459045 ** (-x))

# 生成四维输入
B, n, num_heads, head_dim = 16, 1024, 8, 32
q = torch.randn(B, n, num_heads, head_dim, device="cuda")
k = torch.randn(B, n, num_heads, head_dim, device="cuda")
v = torch.randn(B, n, num_heads, head_dim, device="cuda")
rab = torch.randn(B, n, n, device="cuda")

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


# 记录开始时间
start_event.record()
# 原始计算
qk = torch.einsum("bnhd,bmhd->bhnm", q, k)
attn = torch.einsum("bhnm,bmhd->bnhd", qk, v)

end_event.record()
torch.cuda.synchronize()
print("Time: ", start_event.elapsed_time(end_event))
    

attn1 = hstu_fused_attention(q, k, v, rab,False).permute(0, 2, 1, 3).contiguous()
attn2 = hstu_fused_attention_v2(q, k, v, rab,False).permute(0, 2, 1, 3).contiguous()

print(attn[1,123,7,1])
print(attn1[1,123,7,1])
print(attn2[1,123,7,1])

#经测试，v2显著快于v1与einsum

print("diff 1:",torch.sum(torch.abs(attn - attn1))/(B*n*num_heads*head_dim))
print("diff 2:",torch.sum(torch.abs(attn - attn2))/(B*n*num_heads*head_dim))
# attn = triton_batched_matmul(q, k)

# assert torch.allclose(qk, attn, atol=1e-3)
