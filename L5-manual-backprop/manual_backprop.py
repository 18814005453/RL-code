"""
L5：手写反向传播 — 打破 loss.backward() 黑箱
拿 MLP+BN，forward 拆成原子步骤，一步步手写反向
每步对照 PyTorch autograd 验证梯度一致
"""
import torch
import torch.nn.functional as F
import math

# ===================== 1. 数据 =====================
DATA = '/Users/chenyanji/Desktop/RL-code/L2-makemore/names.txt'
words = open(DATA).read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
vocab_size = len(stoi)

block_size, embedding_dim, hidden_dim, batch_size = 3, 10, 200, 32

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))

def build_dataset(word_list):
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            X.append(context.copy()); Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build_dataset(words[:n1])
g_batch = torch.Generator().manual_seed(42)
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g_batch)
Xb_t, Yb_t = Xtr[ix], Ytr[ix]
n = batch_size
eps = 1e-5

# ===================== 2. 参数（一套，手动和 autograd 共用） =====================
fan_in = block_size * embedding_dim
gain = 5.0 / 3.0
g_p = torch.Generator().manual_seed(2147483647)

C   = torch.randn(vocab_size, embedding_dim, generator=g_p)
W1  = torch.randn(fan_in, hidden_dim, generator=g_p)
b1  = torch.randn(hidden_dim, generator=g_p)
W2  = torch.randn(hidden_dim, vocab_size, generator=g_p)
b2  = torch.randn(vocab_size, generator=g_p)
bngain = torch.randn(hidden_dim, generator=g_p)
bnbias = torch.randn(hidden_dim, generator=g_p)

with torch.no_grad():
    W1.data *= gain / math.sqrt(fan_in)
    b1.data.zero_()
    W2.data *= 0.01
    b2.data.zero_()
    bngain.data.fill_(1.0)
    bnbias.data.zero_()

params = [C, W1, b1, W2, b2, bngain, bnbias]
for p in params:
    p.requires_grad = True

Xb = Xb_t.clone()
Yb = Yb_t.clone()

# ===================== 3. forward pass（原子步骤，每个中间变量保存） =====================
emb    = C[Xb]                                              # ① (32,3,10)
embcat = emb.view(n, -1)                                     # ② (32,30)

hprebn = embcat @ W1 + b1                                    # ③ (32,200)

bnmeani   = hprebn.mean(0, keepdim=True)                      # ④ BN: mean
bndiff    = hprebn - bnmeani                                  # ⑤ BN: 去中心
bndiff2   = bndiff ** 2                                       # ⑥ BN: 平方
bnvar     = bndiff2.sum(0, keepdim=True) / (n - 1)            # ⑦ BN: var
bnvar_inv = 1.0 / torch.sqrt(bnvar + eps)                     # ⑧ BN: 1/σ
bnnorm    = bndiff * bnvar_inv                                # ⑨ BN: norm
hpreact   = bngain * bnnorm + bnbias                          # ⑩ BN: γx̂+β

h = torch.tanh(hpreact)                                       # ⑪ tanh

logits = h @ W2 + b2                                          # ⑫ (32,27)

# softmax 原子步骤
logit_maxes  = logits.max(dim=1, keepdim=True).values
norm_logits  = logits - logit_maxes
counts       = norm_logits.exp()
counts_sum   = counts.sum(dim=1, keepdim=True)
counts_sum_inv = counts_sum ** -1
probs        = counts * counts_sum_inv
log_probs    = probs.log()
losses       = -log_probs[range(n), Yb]
loss         = losses.mean()                                   # ⑬ 标量
print(f"loss = {loss.item():.4f}")

# ===================== 4. PyTorch autograd（对照用） =====================
for p in params:
    p.grad = None
loss.backward()
# 保存 autograd 结果
torch_grads = {name: p.grad.clone() for name, p in
               zip(['C','W1','b1','W2','b2','bngain','bnbias'], params)}

# ===================== 5. 手动反向传播 =====================
def manual_backward():
    """手工反向传播，返回 7 个参数的梯度"""
    # ---- 5.1  loss = mean(losses) ----
    dloss = 1.0
    dlosses = dloss / n * torch.ones(n)                        # mean反向=广播1/n

    # ---- 5.2  losses = -log_probs[range(n), Yb] ----
    d_log_probs = torch.zeros_like(log_probs)
    d_log_probs[range(n), Yb] = -dlosses

    # ---- 5.3  log_probs = log(probs) ----
    d_probs = d_log_probs / probs                              # dlog(x)=1/x

    # ---- 5.4  probs = counts * counts_sum_inv ----
    d_counts_sum_inv = (d_probs * counts).sum(dim=1, keepdim=True)
    d_counts = d_probs * counts_sum_inv

    # ---- 5.5  counts_sum_inv = counts_sum^-1 ----
    d_counts_sum = d_counts_sum_inv * (-1.0 / counts_sum**2)

    # ---- 5.6  counts_sum = counts.sum(dim=1) → sum的反向=广播 ----
    d_counts += d_counts_sum * torch.ones_like(counts)

    # ---- 5.7  counts = exp(norm_logits) ----
    d_norm_logits = d_counts * counts                           # dexp(x)=exp(x)

    # ---- 5.8  norm_logits = logits - logit_maxes ----
    # logit_maxes 不影响，因为 logit_maxes 只平移 → 反向梯度全传给 logits
    d_logits = d_norm_logits.clone()

    # ---- 5.9  logits = h @ W2 + b2 ----
    d_h   = d_logits @ W2.T
    d_W2  = h.T @ d_logits
    d_b2  = d_logits.sum(dim=0)

    # ---- 5.10 h = tanh(hpreact) ----
    d_hpreact = d_h * (1.0 - h**2)                              # dtanh = 1-tanh²

    # ---- 5.11 hpreact = bngain * bnnorm + bnbias ----
    d_bngain = (d_hpreact * bnnorm).sum(dim=0)
    d_bnbias = d_hpreact.sum(dim=0)
    d_bnnorm = d_hpreact * bngain

    # ---- 5.12 bnnorm = bndiff * bnvar_inv ----
    d_bnvar_inv = (d_bnnorm * bndiff).sum(dim=0, keepdim=True)
    d_bndiff = d_bnnorm * bnvar_inv

    # ---- 5.13 bnvar_inv = (bnvar+eps)^(-0.5) ----
    d_bnvar = d_bnvar_inv * (-0.5) * (bnvar + eps)**(-1.5)     # d(x^{-0.5})=-½·x^{-1.5}

    # ---- 5.14 bnvar = sum(bndiff²)/(n-1) → sum反向=广播 ----
    d_bndiff2 = (1.0/(n-1)) * d_bnvar * torch.ones_like(bndiff2)

    # ---- 5.15 bndiff2 = bndiff² ----
    d_bndiff += 2.0 * bndiff * d_bndiff2                       # dsq=2x

    # ---- 5.16 bndiff = hprebn - bnmeani ----
    d_hprebn  = d_bndiff.clone()
    d_bnmeani = -d_bndiff.sum(dim=0, keepdim=True)             # -μ

    # ---- 5.17 bnmeani = hprebn.mean(0) → mean反向=广播1/n ----
    d_hprebn += d_bnmeani / n

    # ---- 5.18 hprebn = embcat @ W1 + b1 ----
    d_embcat = d_hprebn @ W1.T
    d_W1     = embcat.T @ d_hprebn
    d_b1     = d_hprebn.sum(dim=0)

    # ---- 5.19 embcat = emb.view(n,-1) → reshape反向=reshape回去 ----
    d_emb = d_embcat.view(emb.shape)

    # ---- 5.20 emb = C[Xb] → 查表反向=加回原位置 ----
    d_C = torch.zeros_like(C)
    d_C.index_add_(0, Xb.view(-1), d_emb.view(-1, embedding_dim))

    return d_C, d_W1, d_b1, d_W2, d_b2, d_bngain, d_bnbias


m_C, m_W1, m_b1, m_W2, m_b2, m_bngain, m_bnbias = manual_backward()

# ===================== 6. 对照验证 =====================
print("\n========== 7 个参数梯度对照（手动 vs autograd） ==========")
manual_grads = [m_C, m_W1, m_b1, m_W2, m_b2, m_bngain, m_bnbias]
all_ok = True
for name, m_grad, t_grad in zip(['C','W1','b1','W2','b2','bngain','bnbias'],
                                 manual_grads, [torch_grads[n] for n in ['C','W1','b1','W2','b2','bngain','bnbias']]):
    diff = (m_grad - t_grad).abs().max().item()
    ok = diff < 1e-5
    all_ok = all_ok and ok
    mark = "✅" if ok else f"❌ diff={diff:.2e}"
    m_str = f"manual_max={m_grad.abs().max():.4f}" if not ok else ""
    print(f"  {name}: {mark}  {m_str}")

if all_ok:
    print("\n========== ✅ 全部 20 步反向传播，手动和 PyTorch 完全一致 ==========")
else:
    print("\n========== ❌ 有差异，需要检查 ==========")

print("Forward 链路: C→emb→embcat→W1+b1→BN(7步)→tanh→W2+b2→softmax→loss")
print("Backward 链路: loss→log_probs→probs→counts→norm_logits→logits→h→hpreact")
print("               →BNnorm→BNvar_inv→BNvar→BNdiff²→BNdiff→hprebn→W1→embcat→emb→C")
print("\n你现在知道 loss.backward() 里面的每一行在干什么。")
