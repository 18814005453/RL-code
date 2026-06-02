"""
L2 makemore (Bigram) — 完整的统计概率模型
数据集: names.txt (32032英文名) / chinese_names.txt (9523中文名)
目标: 统计 → 概率 → 采样 → 评估

改 USE_CHINESE 切数据集，其他代码不用动
"""
USE_CHINESE = False  # False=英文, True=中文

import torch
import matplotlib.pyplot as plt

# %% 1. 加载数据
DATA = 'chinese_names.txt' if USE_CHINESE else 'names.txt'
words = open(DATA, 'r').read().splitlines()
print(f"数据集: {DATA}, 总名字数: {len(words)}")
print(f"前10个: {words[:10]}")

# %% 2. 建字表
chars = sorted(list(set(''.join(words))))
stoi = {s: i for i, s in enumerate(chars)}
stoi['<S>'] = len(stoi)
stoi['<E>'] = len(stoi)
itos = {i: s for s, i in stoi.items()}  # 反向查：索引 → 字符
vocab_size = len(stoi)
print(f"字表大小: {vocab_size} (0-{vocab_size-2}=字母, {vocab_size-2}=<S>, {vocab_size-1}=<E>)")

# %% 3. 统计计数
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# %% 4. 可视化计数表（需要看的时候取消注释）
# plt.figure(figsize=(16, 16))
# plt.imshow(N, cmap='Blues')
# for i in range(vocab_size):
#     for j in range(vocab_size):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha='center', va='bottom', color='gray', fontsize=6)
#         plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray', fontsize=8)
# plt.axis('off')
# plt.title('Bigram count matrix: brighter = more frequent pairs')
# plt.show()

# %% 5. Demo 1: 简单均匀采样（随机生成 5 个名字）
g = torch.Generator().manual_seed(2147483647)
print("\n=== 均匀采样（随机瞎猜）===")
for _ in range(5):
    out = []
    ix = stoi['<S>']
    while True:
        p = torch.ones(vocab_size) / vocab_size   # 等概率
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == stoi['<E>']:
            break
        out.append(itos[ix])
    print(''.join(out))

# %% 6. 计数 → 概率表
P = N.float()
# 给每行加 1 做平滑：没见过的组合也有一点概率（避免概率为 0）
P += 1
P = P / P.sum(dim=1, keepdim=True)

# %% 7. Demo 2: 基于 bigram 概率采样
g = torch.Generator().manual_seed(2147483647)
print("\n=== Bigram 概率采样 ===")
for _ in range(10):
    out = []
    ix = stoi['<S>']
    while True:
        p = P[ix, :]     # 每一行各列 = 下一字符的概率
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == stoi['<E>']:
            break
        out.append(itos[ix])
    print(''.join(out))

# %% 8. 评估模型 —— 负对数似然（NLL）
# 对每个名字里的每对字符：log P(下一字符|当前字符)，求和取负
log_likelihood = 0.0
n_pairs = 0

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        log_likelihood += torch.log(prob)
        n_pairs += 1

nll = -log_likelihood / n_pairs     # 平均负对数似然
print(f"\n=== 模型评估 ===")
print(f"总字符对数: {n_pairs}")
print(f"负对数似然 (NLL): {nll:.4f}")
print(f"等价困惑度 (perplexity): {torch.exp(nll):.4f}")
# NLL ≈ 2.4 左右（英文），越低越好
# 意义：模型预测下一个字符时，平均需要 2.4 个 bit 来确定答案

# %% 9. 【桥梁】神经网络 bigram —— 梯度下降学出和计数表一样的结果
import torch.nn as nn
from torch.nn import functional as F

# 把数据转成 PyTorch 的 tensor
xs, ys = [], []
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(f"\n总训练样本: {len(xs)}（每对相邻字符 = 一个样本）")

# 模型：一张 vocab_size × vocab_size 的 logits 表
device = 'cpu'  # 本机CPU就够
logits = nn.Parameter(torch.randn((vocab_size, vocab_size), device=device))
xs, ys = xs.to(device), ys.to(device)
# 初始值是随机的，后面梯度下降会把它调到和 N 表一样

lr = 50
for epoch in range(200):
    logits_batch = logits[xs]
    loss = F.cross_entropy(logits_batch, ys)

    logits.grad = None
    loss.backward()
    logits.data -= lr * logits.grad

    if epoch % 40 == 0:
        print(f"  epoch {epoch:3d}: loss = {loss.item():.4f}")

print(f"  最终 loss = {loss.item():.4f}（计数表: NLL={nll:.4f}）")

# 验证：梯度下降学出的概率表 vs 原始计数统计表（不加平滑）
raw_P = N.float()
row_sum = raw_P.sum(dim=1, keepdim=True)
row_sum[row_sum == 0] = 1  # 全是 0 的行给它变成 1，避免除 0
raw_P = raw_P / row_sum
nn_P = torch.softmax(logits, dim=1).detach().cpu()
# 只看有数据的行（行总数 > 0）
valid = N.sum(dim=1) > 0
diff = (nn_P[valid, :] - raw_P[valid, :].cpu()).abs().max().item()
print(f"  原始计数 vs 梯度下降 → 有效行最大差异: {diff:.6f}（应该 ≈ 0）")
print(f"  结论：梯度下降学出来的就是计数表除以行总数！")

# 用神经网络版 bigram 生成名字
g = torch.Generator().manual_seed(2147483647)
print("\n=== 神经网络 Bigram 生成的名字 ===")
for _ in range(10):
    out = []
    ix = stoi['<S>']
    while True:
        p = nn_P[ix, :]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == stoi['<E>']:
            break
        out.append(itos[ix])
    print(''.join(out))
