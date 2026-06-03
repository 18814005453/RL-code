"""
L4 实验4（补充）：更新/参数比率诊断
Karpathy 最后教的诊断工具 — 检查 lr 是不是太猛或太弱
"""
import torch
import torch.nn.functional as F
import math

# ===================== 数据 =====================
DATA = '/Users/chenyanji/Desktop/RL-code/视频课/makemore/names.txt'
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

# ===================== 训练 + 每 N 步诊断一次 =====================
def train_with_diagnostics(lr=0.1, steps=10000, label=""):
    g = torch.Generator().manual_seed(2147483647)
    fan_in = block_size * embedding_dim
    gain = 5.0 / 3.0

    C  = torch.empty(vocab_size, embedding_dim); C.data.normal_(generator=g); C.requires_grad = True
    W1 = torch.empty(fan_in, hidden_dim)
    W1.data.normal_(generator=g).mul_(gain / math.sqrt(fan_in))
    b1 = torch.empty(hidden_dim); b1.data.zero_()
    W2 = torch.empty(hidden_dim, vocab_size)
    W2.data.normal_(generator=g).mul_(0.01)
    b2 = torch.empty(vocab_size); b2.data.zero_()
    params = [C, W1, b1, W2, b2]
    for p in params:
        p.requires_grad = True

    # 记录每个参数的 update/param 比率历史
    param_names = ['C', 'W1', 'b1', 'W2', 'b2']
    ratio_history = {name: [] for name in param_names}

    losses = []
    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

        emb = C[Xtr[ix]].view(-1, fan_in)
        h = torch.tanh(emb @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, Ytr[ix])

        for p in params:
            p.grad = None
        loss.backward()

        lr_step = 0.1 if step < steps//2 else 0.01
        for p in params:
            if p.grad is not None:
                p.data += -lr_step * p.grad

        # ====== 每 5000 步诊断一次 ======
        if step % 5000 == 0:
            ratios_line = f"[{label}] step {step:5d}: "
            for name, p in zip(param_names, params):
                update_std = (lr_step * p.grad).std()
                param_std  = p.data.std()
                ratio = (update_std / param_std).item()
                ratio_history[name].append(ratio)
                # 标注危险区域
                flag = ""
                if ratio > 1e-2:   flag = "⚠️太大"
                elif ratio < 1e-4: flag = "⚠️太小"
                elif ratio < 1e-5: flag = "❌没更新"
                ratios_line += f"{name}={ratio:.1e}{flag}  "
            print(ratios_line)

        losses.append(loss.item())

    return losses, ratio_history


# ===================== 对比两种 lr =====================
print("========== lr=0.1 诊断 ==========")
losses_01, ratios_01 = train_with_diagnostics(lr=0.1, steps=30000, label="lr=0.1")

print("\n========== lr=0.01 诊断 ==========")
losses_001, ratios_001 = train_with_diagnostics(lr=0.01, steps=30000, label="lr=0.01")

print("\n========== 诊断结论 ==========")
print("理想范围: 10⁻³ (即 0.001) — Karpathy 原话: '大约 -3 在对数坐标上'")
print("> 10⁻²: lr 太大，参数在震荡，train loss 剧烈抖动")
print("< 10⁻⁴: lr 太小，参数几乎不动，train loss 不降")
print("= 10⁻³: 刚好")
print()
for name in ['C', 'W1', 'W2']:
    r01 = ratios_01[name][-1]
    r001 = ratios_001[name][-1]
    print(f"{name}: lr=0.1 → ratio={r01:.1e} | lr=0.01 → ratio={r001:.1e}")

print("\n这些数字比你盯着 loss 曲线更早发现问题——")
print("如果 loss 不降但你不知道是 lr 太小还是初始化有问题，看这个比例就知道了。")
print("DAPO/GRPO/PPO 训练时每个 epoch 打一次这个比例，调参前不用猜。")
