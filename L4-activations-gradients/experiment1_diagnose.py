"""
L4 实验1：诊断初始化 — 只看 forward，不训练
目标：亲眼看到"randn 初始化 → tanh 全饱和 → 梯度死掉"
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ===================== 数据准备（和 L3 一样） =====================
words = open('/Users/chenyanji/Desktop/RL-code/视频课/makemore/names.txt').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)                          # 27

# 造一个随机 batch（模拟训练第一步的输入）
block_size = 3
g = torch.Generator().manual_seed(42)
batch_size = 64
X = torch.randint(0, vocab_size, (batch_size, block_size), generator=g)

embedding_dim = 10
hidden_dim   = 200

# ===================== 诊断函数 =====================
@torch.no_grad()
def diagnose(name, C, W1, b1):
    emb = C[X]                                          # (64, 3, 10)
    x = emb.view(-1, block_size * embedding_dim)        # (64, 30)  ← 拼成30维向量
    h_preact = x @ W1 + b1                              # (64, 200) ← tanh之前的线性输出
    h = torch.tanh(h_preact)                            # (64, 200) ← 激活后的隐藏层

    # 三个诊断指标
    mean_h = h.mean().item()          # 激活值均值（理想 ~0）
    std_h  = h.std().item()           # 激活值标准差（理想 ~1）
    sat    = (h.abs() > 0.97).float().mean().item()  # 饱和率（理想 <5%）

    # 找 dead neuron：这个神经元对所有64个样本都饱和了
    dead = ((h.abs() > 0.97).all(dim=0)).sum().item()

    print(f"{name:>18}:  mean={mean_h:+.3f}  std={std_h:.3f}  "
          f"饱和率={sat*100:5.1f}%  dead neurons={dead:3d}/{hidden_dim}")

    return h, h_preact

# ===================== 对比 3 种初始化 =====================

# A. 纯 randn — 最烂，没有任何归一化
C  = torch.randn(vocab_size, embedding_dim)
W1 = torch.randn(block_size * embedding_dim, hidden_dim)
b1 = torch.randn(hidden_dim) * 0.01
print("========== 初始化对比 ==========")
diagnose("A. randn 裸奔", C, W1, b1)

# B. W1 缩小 — L3 初版的做法，拍脑袋乘了个 0.01
C  = torch.randn(vocab_size, embedding_dim)
W1 = torch.randn(block_size * embedding_dim, hidden_dim) * 0.01
b1 = torch.randn(hidden_dim) * 0
diagnose("B. W1*0.01 盲调", C, W1, b1)

# C. Kaiming 初始化 — 用增益/√fan_in 精确压 std
fan_in = block_size * embedding_dim  # 30
gain = 5.0 / 3.0                     # tanh 的推荐增益
C  = torch.randn(vocab_size, embedding_dim)
W1 = torch.randn(block_size * embedding_dim, hidden_dim) * (gain / (fan_in ** 0.5))
b1 = torch.randn(hidden_dim) * 0
diagnose("C. Kaiming (gain/√fan_in)", C, W1, b1)

# ===================== 为什么要除√(fan_in) — 矩阵乘法的方差放大 =====================
print("\n========== 矩阵乘法的方差放大效应 ==========")
for fan_in_test in [30, 300, 3000]:
    x_test = torch.randn(1000, fan_in_test)          # 输入 std=1
    W_test = torch.randn(fan_in_test, 200)           # 权重 std=1
    y_test = x_test @ W_test                          # 矩阵乘法
    print(f"fan_in={fan_in_test:4d}:  输入std=1.0  →  输出std={y_test.std().item():.2f}  "
          f"(√fan_in={fan_in_test**0.5:.1f})")
print("结论: 输出std ≈ √fan_in。fan_in越大，放大越猛 → 不除以√fan_in, tanh全饱和")

# ===================== 可视化：三种初始化的 h 分布 =====================
h_randn,   _ = diagnose("", torch.randn(27, 10), torch.randn(30, 200), torch.randn(200)*0.01)
h_kaiming, _ = diagnose("", torch.randn(27, 10),
                         torch.randn(30, 200) * (5/3)/(30**0.5),
                         torch.randn(200)*0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(h_randn.flatten().numpy(), bins=50, alpha=0.7, label='randn')
axes[0].set_title(f'randn 初始化\nmean={h_randn.mean():.3f}, std={h_randn.std():.3f}')
axes[0].axvspan(-1, -0.97, color='red', alpha=0.1)
axes[0].axvspan(0.97, 1, color='red', alpha=0.1)

axes[1].hist(h_kaiming.flatten().numpy(), bins=50, alpha=0.7, color='green', label='kaiming')
axes[1].set_title(f'Kaiming 初始化\nmean={h_kaiming.mean():.3f}, std={h_kaiming.std():.3f}')
axes[1].axvspan(-1, -0.97, color='red', alpha=0.1)
axes[1].axvspan(0.97, 1, color='red', alpha=0.1)

for ax in axes:
    ax.set_xlabel('激活值 (tanh输出)')
    ax.set_ylabel('频数')
    ax.axvline(0, color='gray', linestyle='--')

plt.suptitle('隐藏层激活值分布对比：randn vs Kaiming（红色区域=饱和区）', fontsize=13)
plt.tight_layout()
plt.savefig('/Users/chenyanji/Desktop/RL-code/L4-activations-gradients/diagnosis_activation.png', dpi=150)
plt.close()
print("\n图片已保存: diagnosis_activation.png")
print("randn:  大量激活值落入红色饱和区 → tanh梯度≈0 → 训练不动")
print("Kaiming: 激活值集中在中间 → tanh梯度有效 → 训练能进行")
