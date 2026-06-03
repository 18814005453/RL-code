"""
L4 实验2：手写 BatchNorm1d + 塞进 MLP
验证：BN 确实能把任意输入当场拉回 mean=0、std=1
对比：有 BN vs 无 BN 的训练曲线
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

# ===================== 数据准备（和 L3 一样） =====================
DATA = '/Users/chenyanji/Desktop/RL-code/视频课/makemore/names.txt'
words = open(DATA).read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(stoi)  # 27

block_size, embedding_dim, hidden_dim, batch_size = 3, 10, 200, 32

# train/dev/test 拆分
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

def build_dataset(word_list):
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            X.append(context.copy())
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte   = build_dataset(words[n2:])

# ===================== 手写 BatchNorm1d =====================
class BatchNorm1d:
    """从零实现：训练用 batch 统计量，推理用 running 统计量"""
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # 可学习参数
        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta  = torch.zeros(dim, requires_grad=True)

        # 推理用的累积统计量
        self.running_mean = torch.zeros(dim)
        self.running_var  = torch.ones(dim)

    def parameters(self):
        return [self.gamma, self.beta]

    def calibrate(self, X_full):
        """训练完成后，在全量验证集上重新计算精准的 running mean/std。
        Karpathy 做法：训练中的滑动平均受限于小 batch（32），
        全量跑一次拿到更准确的全局统计量，推理时用这套。"""
        self.training = True  # 不开训练模式拿不到正确的 mean
        # 分 batch 累积统计量（防止全量数据 OOM）
        batch_size = 256
        means, vars = [], []
        for i in range(0, X_full.shape[0], batch_size):
            x = X_full[i:i+batch_size]
            means.append(x.mean(dim=0))
            vars.append(x.var(dim=0, unbiased=False))
        # 取平均
        self.running_mean = torch.stack(means).mean(dim=0)
        self.running_var  = torch.stack(vars).mean(dim=0)
        self.training = False  # 切回推理模式
        print(f"  校准完成: running_mean avg={self.running_mean.mean().item():.3f}, "
              f"running_var avg={self.running_var.mean().item():.3f}")

    def forward(self, x):
        # x: (batch, dim) — 32个样本 × 200维
        if self.training:
            mean = x.mean(dim=0, keepdim=False)           # (dim,)
            var  = x.var(dim=0, keepdim=False, unbiased=False)  # (dim,) 有偏估计
        else:
            mean = self.running_mean
            var  = self.running_var

        # 归一化
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.training:
            # 更新 running 统计量（指数滑动平均）
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var

        return self.gamma * x_hat + self.beta


# ===================== 训练函数（支持 BN 开关） =====================
@torch.no_grad()
def evaluate(params, bns, X_eval, Y_eval):
    """在评估集上算 loss，推理模式（用 running stats）"""
    for bn in bns:
        bn.training = False

    C, W1, b1, W2, b2 = params
    emb = C[X_eval].view(-1, block_size * embedding_dim)

    h = emb @ W1 + b1
    for bn in bns:
        h = bn.forward(h)
    h = torch.tanh(h)

    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y_eval)

    for bn in bns:
        bn.training = True
    return loss.item()


def train(use_bn=False, steps=50000, label=""):
    g = torch.Generator().manual_seed(2147483647)
    fan_in = block_size * embedding_dim    # 30
    gain = 5.0 / 3.0

    # 先创建叶子张量，再用 .data 原地修改（保持叶子属性）
    C  = torch.empty(vocab_size, embedding_dim)
    W1 = torch.empty(fan_in, hidden_dim)
    b1 = torch.empty(hidden_dim)
    W2 = torch.empty(hidden_dim, vocab_size)
    b2 = torch.empty(vocab_size)
    # 从 generator 手动采样填入（确保叶子属性且 requires_grad 生效）
    C.data.normal_(generator=g)
    W1.data.normal_(generator=g).mul_(gain / math.sqrt(fan_in))
    b1.data.zero_()
    W2.data.normal_(generator=g).mul_(0.01)
    b2.data.zero_()
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True

    # 如果是 BN 版本，在 tanh 前面插入 BN
    bn = BatchNorm1d(hidden_dim) if use_bn else None
    bns = [bn] if bn else []

    losses = []
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        # forward
        emb = C[Xtr[ix]].view(-1, fan_in)
        h = emb @ W1 + b1
        if bn:
            h = bn.forward(h)
        h_preact = h
        h = torch.tanh(h)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        # backward
        all_p = params + [p for bn_obj in bns for p in bn_obj.parameters()]
        for p in all_p:
            p.grad = None
        loss.backward()

        lr = 0.1 if step < steps//2 else 0.01
        for p in all_p:
            if p.grad is not None:
                p.data += -lr * p.grad

        losses.append(loss.item())

        if step % 10000 == 0:
            dev_l = evaluate(params, bns, Xdev, Ydev)
            print(f"[{label}] step {step:5d}: train loss {loss.item():.3f}, dev loss {dev_l:.3f}")

    dev_l = evaluate(params, bns, Xdev, Ydev)
    print(f"[{label}] 最终: dev loss {dev_l:.3f}")
    return losses, bn


# ===================== 1. 先不做训练，验证 BN 的归一化效果 =====================
print("========== 先验证 BN 的归一化效果 ==========")
g = torch.Generator().manual_seed(42)
X_test = torch.randn(batch_size, hidden_dim) * 2.0 + 3.0  # 制造一个有偏移的输入：std~2, mean~3
print(f"BN 之前: mean={X_test.mean():.2f}, std={X_test.std():.2f}")

bn_test = BatchNorm1d(hidden_dim)
out_test = bn_test.forward(X_test)
print(f"BN 之后: mean={out_test.mean():.2f}, std={out_test.std():.2f}")
print("gamma 全 1、beta 全 0 → 强制输出 mean≈0, std≈1\n")

# ===================== 2. 训练对比：无 BN vs 有 BN =====================
print("========== 训练对比：无 BN vs 有 BN ==========")
losses_no_bn, _    = train(use_bn=False, steps=50000, label="无BN")
losses_with_bn, bn_final = train(use_bn=True,  steps=50000, label="有BN")

# ===================== 3. 可视化对比 =====================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(losses_no_bn, alpha=0.7, label='无 BN')
plt.plot(losses_with_bn, alpha=0.7, label='有 BN')
plt.axvline(x=25000, color='orange', linestyle='--', alpha=0.5, label='lr 0.1→0.01')
plt.xlabel('step')
plt.ylabel('train loss')
plt.title('训练曲线对比：无 BN vs 有 BN')
plt.legend()

plt.subplot(1, 2, 2)
if bn_final:
    plt.bar(range(hidden_dim), bn_final.gamma.detach().numpy(), alpha=0.7, label='γ')
    plt.axhline(y=1.0, color='gray', linestyle='--', label='初始 γ=1')
    plt.xlabel('神经元编号')
    plt.ylabel('γ 值')
    plt.title('训练后每个神经元的 γ（学出来的最优分布宽度）')
    plt.legend()

plt.tight_layout()
plt.savefig('/Users/chenyanji/Desktop/RL-code/L4-activations-gradients/batchnorm_comparison.png', dpi=150)
plt.close()
print("\n图片已保存: batchnorm_comparison.png")
print("左图：无BN的loss下降更抖动，有BN的更稳定")
print("右图：γ偏离1的神经元→BN学出了非标准正态分布，证明γ/β不是摆设")
