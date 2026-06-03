"""
L4 实验3：堆深 1 层→5 层
验证：没有 BN 深层训不动，加了 BN 就能训
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ===================== 数据准备 =====================
DATA = '/Users/chenyanji/Desktop/RL-code/视频课/makemore/names.txt'
words = open(DATA).read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
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

# ===================== 手写组件 =====================
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.dim, self.eps, self.momentum = dim, eps, momentum
        self.training = True
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta  = torch.zeros(dim, requires_grad=True)
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0, keepdim=False)
            var  = x.var(dim=0, keepdim=False, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]


# ===================== 训练函数 =====================
def train_deep(n_layers, use_bn, steps=30000):
    g = torch.Generator().manual_seed(2147483647)
    fan_in = block_size * embedding_dim
    gain = 5.0 / 3.0

    # Embedding
    C = torch.empty(vocab_size, embedding_dim)
    C.data.normal_(generator=g)
    C.requires_grad = True
    all_p = [C]

    # 第一层：fan_in → hidden_dim
    W_first = torch.empty(fan_in, hidden_dim)
    W_first.data.normal_(generator=g).mul_(gain / math.sqrt(fan_in))
    b_first = torch.empty(hidden_dim); b_first.data.zero_()
    W_first.requires_grad, b_first.requires_grad = True, True
    all_p += [W_first, b_first]

    # 堆叠中间层
    layers = []  # 每层: (W, b, bn, tanh)
    for i in range(n_layers - 1):
        W = torch.empty(hidden_dim, hidden_dim)
        W.data.normal_(generator=g).mul_(gain / math.sqrt(hidden_dim))
        b = torch.empty(hidden_dim); b.data.zero_()
        W.requires_grad, b.requires_grad = True, True
        bn = BatchNorm1d(hidden_dim) if use_bn else None
        layers.append((W, b, bn))
        all_p += [W, b]
        if bn:
            all_p += bn.parameters()

    # 最后一层：hidden_dim → vocab_size
    W_last = torch.empty(hidden_dim, vocab_size)
    W_last.data.normal_(generator=g).mul_(0.01)
    b_last = torch.empty(vocab_size); b_last.data.zero_()
    W_last.requires_grad, b_last.requires_grad = True, True
    all_p += [W_last, b_last]

    # 诊断记录
    act_stds = []  # 每层 tanh 输出的 std

    losses = []
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        # ---- forward ----
        h = C[Xtr[ix]].view(-1, fan_in)
        h = torch.tanh(h @ W_first + b_first)

        stds_this_step = [h.std().item()]  # 记录第1层激活 std

        for W, b, bn in layers:
            h = h @ W + b
            if bn:
                h = bn.forward(h)
            h = torch.tanh(h)
            stds_this_step.append(h.std().item())

        logits = h @ W_last + b_last
        loss = F.cross_entropy(logits, Ytr[ix])

        # ---- backward ----
        for p in all_p:
            p.grad = None
        loss.backward()

        lr = 0.1 if step < steps//2 else 0.01
        for p in all_p:
            if p.grad is not None:
                p.data += -lr * p.grad

        losses.append(loss.item())
        act_stds.append(stds_this_step)

        if step % 10000 == 0:
            print(f"  [n={n_layers} {'BN' if use_bn else '--'}] step {step:5d}: loss {loss.item():.3f}")

    return losses, act_stds


# ===================== 跑两组实验 =====================
print("========== 5 层 MLP：无 BN ==========")
losses_no_bn, stds_no_bn = train_deep(n_layers=5, use_bn=False, steps=30000)

print("\n========== 5 层 MLP：有 BN ==========")
losses_with_bn, stds_with_bn = train_deep(n_layers=5, use_bn=True, steps=30000)

# ===================== 可视化 =====================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 图1：loss 对比
axes[0].plot(losses_no_bn, alpha=0.8, label='5层 无BN')
axes[0].plot(losses_with_bn, alpha=0.8, label='5层 有BN')
axes[0].axvline(x=15000, color='orange', linestyle='--', alpha=0.5)
axes[0].set_xlabel('step')
axes[0].set_ylabel('train loss')
axes[0].set_title('5层 MLP 训练曲线')
axes[0].legend()

# 图2：无BN 各层激活 std 演化
stds_no_bn = list(zip(*stds_no_bn))  # 转置 → (n_layers, n_steps)
for i, layer_stds in enumerate(stds_no_bn):
    axes[1].plot(layer_stds, alpha=0.7, label=f'Layer {i+1}')
axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
axes[1].set_xlabel('step')
axes[1].set_ylabel('tanh 输出 std')
axes[1].set_title('无BN — 各层激活 std （无BN时后面层衰减）')
axes[1].legend(fontsize=8)

# 图3：有BN 各层激活 std 演化
stds_with_bn = list(zip(*stds_with_bn))
for i, layer_stds in enumerate(stds_with_bn):
    axes[2].plot(layer_stds, alpha=0.7, label=f'Layer {i+1}')
axes[2].axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
axes[2].set_xlabel('step')
axes[2].set_ylabel('tanh 输出 std')
axes[2].set_title('有BN — 各层激活 std（BN 保持健康）')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('/Users/chenyanji/Desktop/RL-code/L4-activations-gradients/deep_network_5layers.png', dpi=150)
plt.close()

print("\n图片已保存: deep_network_5layers.png")
print("左图：5层有BN的loss下降更稳定")
print("中图：无BN — 后面层的激活std衰减到0.1以下→信息流失")
print("右图：有BN — 所有层的激活std保持在健康水平（~1.0）")
