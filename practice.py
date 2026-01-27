import torch
import torch.nn.functional as F
from llmlib import cfg

torch.manual_seed(1337)

B, T, C = cfg.BATCH_SIZE, cfg.CONTEXT_LEN, 2
x = torch.randn(B, T, C)
print(x.shape)
# inefficient way of avg-attention
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #from this batch t,C
        xbow[b, t] = torch.mean(xprev, 0) #avg over t for C

# efficient way of avg-attention:
torch.manual_seed(42)
a = torch.tril(torch.ones(3,3))
print(torch.sum(a, dim=1, keepdim=True))
a = a / torch.sum(a, dim=1, keepdim=True)
b = torch.randint(0,10,(3,2)).float()
c = a @ b
print(a)
print(b)
print(c)

# Doing it for T:
weights = torch.tril(torch.ones(cfg.CONTEXT_LEN, cfg.CONTEXT_LEN))
weights = weights / torch.sum(weights, dim=1, keepdim=True)
print(weights)
xbow2 = weights @ x # (B, T, T) @ (B, T, C) ---> (B, T, C), @ is batched matmul in parallel, creates B for weights.
                    # then it is simply (T, T) @ (T, C).
print(xbow)
print(xbow2)
print(torch.allclose(xbow, xbow2))

print(xbow - xbow2)
# 3rd version:
tril = torch.tril(torch.ones(cfg.CONTEXT_LEN, cfg.CONTEXT_LEN))
wei = torch.zeros(cfg.CONTEXT_LEN, cfg.CONTEXT_LEN)
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
print(torch.allclose(xbow2, xbow3))


