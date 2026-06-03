"""
L3 makemore (MLP) — 跟着 Karpathy 视频手写
用 MLP 替代 bigram 计数表，预测下一个字符

Bengio et al. 2003: A Neural Probabilistic Language Model
"""
DO_GRID_SEARCH = False   # True = 超参数搜索, False = 单次训练

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ==============================
# 1. 加载数据 + 建字表
# ==============================
DATA = 'names.txt'
words = open(DATA, 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)
print(f"字表大小: {vocab_size}, 名字数: {len(words)}")

# ==============================
# 2. 构建所有样本 + train/dev/test 拆分
# ==============================
block_size = 3

def build_dataset(word_list):
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context.copy())
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte   = build_dataset(words[n2:])
print(f"train: {Xtr.shape[0]}, dev: {Xdev.shape[0]}, test: {Xte.shape[0]}")

# ==============================
# 3. 训练函数（超参数作为参数传入）
# ==============================
@torch.no_grad()
def eval_loss(C, W1, b1, W2, B2, X_eval, Y_eval, batch_size=32):
    total_loss = 0.0
    total_n = 0
    for i in range(0, X_eval.shape[0], batch_size):
        ix_slice = slice(i, i + batch_size)
        emb = C[X_eval[ix_slice]]
        h = torch.tanh(emb.view(-1, block_size * C.shape[1]) @ W1 + b1)
        logits = h @ W2 + B2
        loss = F.cross_entropy(logits, Y_eval[ix_slice])
        total_loss += loss.item() * (ix_slice.stop - ix_slice.start)
        total_n += (ix_slice.stop - ix_slice.start)
    return total_loss / total_n


def train(embedding_dim=10, hidden_dim=200, steps=200000, lr_start=0.1, lr_end=0.01, verbose=True):
    """训练一个 MLP，返回 (train_loss, dev_loss, parameters_dict)"""
    g = torch.Generator().manual_seed(2147483647)

    # 初始化参数
    C  = torch.randn((vocab_size, embedding_dim), generator=g)
    W1 = torch.randn((block_size * embedding_dim, hidden_dim), generator=g) * (5/3) / (block_size * embedding_dim)**0.5
    b1 = torch.randn(hidden_dim, generator=g) * 0
    W2 = torch.randn((hidden_dim, vocab_size), generator=g) * 0.01
    B2 = torch.randn(vocab_size, generator=g) * 0

    params = [C, W1, b1, W2, B2]
    for p in params:
        p.requires_grad = True

    n_params = sum(p.nelement() for p in params)

    # 训练循环
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)

        emb = C[Xtr[ix]]
        h = torch.tanh(emb.view(-1, block_size * embedding_dim) @ W1 + b1)
        logits = h @ W2 + B2
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in params:
            p.grad = None
        loss.backward()

        lr = lr_start if step < steps // 2 else lr_end
        for p in params:
            p.data += -lr * p.grad

    train_loss = eval_loss(C, W1, b1, W2, B2, Xtr, Ytr)
    dev_loss   = eval_loss(C, W1, b1, W2, B2, Xdev, Ydev)

    if verbose:
        print(f"  emb={embedding_dim:2d}, hidden={hidden_dim:3d}, "
              f"params={n_params:>6,}, train={train_loss:.3f}, dev={dev_loss:.3f}")

    return train_loss, dev_loss, n_params, (C, W1, b1, W2, B2)


# ==============================
# 4. 超参数搜索 或 单次训练
# ==============================
if DO_GRID_SEARCH:
    # Karpathy 原版做法：拿 dev loss 选参数，不碰 test set
    print("\n========== 超参数搜索 (用 dev loss 选最优) ==========")
    results = []
    for emb_dim in [2, 5, 10, 20]:
        for hid_dim in [50, 100, 200, 300]:
            train_l, dev_l, n_param, _ = train(
                embedding_dim=emb_dim, hidden_dim=hid_dim,
                steps=200000, verbose=True
            )
            results.append((dev_l, train_l, n_param, emb_dim, hid_dim))

    # 按 dev loss 排序
    results.sort()
    print("\n========== 搜索结果（按 dev loss 排序）==========")
    print(f"{'embed':>6} {'hidden':>7} {'params':>8} {'train':>7} {'dev':>7}")
    for dev_l, train_l, n, e, h in results:
        marker = " ← 最优" if (e, h) == (results[0][3], results[0][4]) else ""
        print(f"{e:>6} {h:>7} {n:>8,} {train_l:>7.3f} {dev_l:>7.3f}{marker}")

    # 用最优超参重新训练（开 verbose），然后再在 test 上评估
    best_emb, best_hid = results[0][3], results[0][4]
    print(f"\n========== 最优超参: emb={best_emb}, hidden={best_hid} ==========")
    best_params = train(embedding_dim=best_emb, hidden_dim=best_hid, steps=200000, verbose=False)
    train_loss, dev_loss, n_param, (C, W1, b1, W2, B2) = best_params
    test_loss = eval_loss(C, W1, b1, W2, B2, Xte, Yte)
    print(f"\n最优模型最终: train={train_loss:.3f}, dev={dev_loss:.3f}, test={test_loss:.3f}, params={n_param:,}")

else:
    # 单次训练
    train_loss, dev_loss, n_param, (C, W1, b1, W2, B2) = train(
        embedding_dim=10, hidden_dim=200, steps=200000, verbose=False
    )
    test_loss = eval_loss(C, W1, b1, W2, B2, Xte, Yte)
    print(f"\n========== 最终评估 ==========")
    print(f"train: {train_loss:.3f}, dev: {dev_loss:.3f}, test: {test_loss:.3f}, params: {n_param:,}")

# ==============================
# 5. 采样生成
# ==============================
print("\n========== 采样生成 ==========")
g = torch.Generator().manual_seed(2147483647 + 1)  # 换个种子，不是训练用的
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        emb_dim = C.shape[1]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + B2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    print(''.join(out))

print(f"\nL2 bigram 对照: NLL ≈ 2.45 (729 参数)")
print(f"L3 MLP 对照:   test NLL = {test_loss:.2f} ({n_param:,} 参数)")
