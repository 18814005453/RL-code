"""
L6：WaveNet 层级架构 — makemore Part 5
核心思想：不一次性压扁所有输入字符，而是树状渐进融合
  第1层：相邻2字符合并 → Bigram
  第2层：相邻2个Bigram合并 → 4-gram
  第3层：相邻2个4-gram合并 → 8-gram → 预测输出

与 L3/L4/L5 的关键差异：
  - block_size 3→8（感受野翻倍不止）
  - 去掉 Flatten，用 FlattenConsecutive(2) 每次只合并相邻两个
  - BatchNorm1D 需要支持 3D 张量 (N, T, C)
"""
import torch
import torch.nn.functional as F
import random

# ==================== 1. 数据准备 ====================
# 数据文件：与本文件同目录下的 names.txt
import os
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'names.txt')
words = open(DATA).read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {c: i+1 for i, c in enumerate(chars)}
stoi['.'] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(stoi)
print(f"vocab_size = {vocab_size}, 总词数 = {len(words)}")

block_size = 8  # 🔑 从3改成8，现在每个样本用8个字符预测第9个

def build_dataset(word_list):
    """构建数据集：每个词拆成多个 (context[8], target) 样本"""
    X, Y = [], []
    for w in word_list:
        context = [0] * block_size
        for ch in w + '.':
            X.append(context.copy())
            Y.append(stoi[ch])
            context = context[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f"Xtr shape: {Xtr.shape}, Ytr shape: {Ytr.shape}")
print(f"Xdev shape: {Xdev.shape}, Yte shape: {Yte.shape}")

# ==================== 2. 自定义层 ====================

class FlattenConsecutive:
    """
    将连续的 n 个时间步合并为一个。
    输入: (B, T, C)
    输出: (B, T//n, C*n)  — 如果 T//n==1 则 squeeze 到 (B, C*n)

    举例：输入 (32, 8, 10)，n=2
         → view(32, 4, 20)  # 每2个相邻字符合并，通道翻倍
    """
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:  # 只剩1个时间步，压掉中间维度
            x = x.squeeze(1)
        return x

    def parameters(self):
        return []

class BatchNorm1D:
    """
    支持 2D 和 3D 输入的 BatchNorm。

    关键改动（vs L4）：
    - 2D (N, C):      在 dim=0 (batch) 上统计           → C 个统计量
    - 3D (N, T, C):   在 dim=(0,1) (batch+time) 上统计  → 仍然是 C 个统计量

    为什么3D要在 batch+time 上一起统计？
    如果只在 batch 上统计 (dim=0)，每个 time step 独立统计，
    batch_size=32 时每个 time step 只有 32 个样本 → 方差估计不稳定。
    在 (0,1) 上统计 → 32*8=256 个样本 → 稳定的方差估计。
    """
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.training = True

        self.gamma = torch.ones(dim, requires_grad=True)
        self.beta = torch.zeros(dim, requires_grad=True)
        # running stats 不需要梯度
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0           # (N, C) → 在 batch 上统计
            else:  # x.ndim == 3
                dim = (0, 1)      # (N, T, C) → 在 batch+time 上统计

            batch_mean = x.mean(dim=dim)
            batch_var = x.var(dim=dim, correction=0)  # correction=0: 除以n（与视频一致）

            # 指数移动平均更新 running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_hat + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ==================== 3. 构建层级化模型 ====================
n_emb = 24      # 嵌入维度（L3用10，这里提高到24）
n_hidden = 68   # 隐藏层维度（L3用200，这里降到68——因为层级化让每层更"轻"）

model = Sequential([
    # Layer 0: 字符 → 嵌入向量
    torch.nn.Embedding(vocab_size, n_emb),                    # (B, 8) → (B, 8, 24)

    # Layer 1: Bigram 层 — 相邻2个字符 → 2-gram
    FlattenConsecutive(2),                                    # (B, 8, 24) → (B, 4, 48)
    torch.nn.Linear(n_emb * 2, n_hidden),                     # (B, 4, 48) → (B, 4, 68)
    BatchNorm1D(n_hidden),
    torch.nn.Tanh(),

    # Layer 2: 4-gram 层 — 相邻2个Bigram → 4-gram
    FlattenConsecutive(2),                                    # (B, 4, 68) → (B, 2, 136)
    torch.nn.Linear(n_hidden * 2, n_hidden),                  # (B, 2, 136) → (B, 2, 68)
    BatchNorm1D(n_hidden),
    torch.nn.Tanh(),

    # Layer 3: 8-gram 层 — 相邻2个4-gram → 8-gram → 输出
    FlattenConsecutive(2),                                    # (B, 2, 68) → (B, 1, 136) → squeeze → (B, 136)
    torch.nn.Linear(n_hidden * 2, vocab_size),                # (B, 136) → (B, 27)
])

# Kaiming 初始化，gain=5/3 给 tanh
fan_in = n_emb * 2
gain = 5.0 / 3.0
with torch.no_grad():
    for layer in model.layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data *= gain / (fan_in ** 0.5)
            # 后续层的 fan_in 动态更新（简化处理：不再逐层调整）
        if isinstance(layer, torch.nn.Embedding):
            layer.weight.data *= 0.1  # 小嵌入初始值

# 统计参数
n_params = sum(p.nelement() for p in model.parameters())
print(f"\n模型参数: {n_params:,}")
print(f"架构: Embed({vocab_size},{n_emb}) → Bigram(48→{n_hidden}) → 4-gram({n_hidden*2}→{n_hidden}) → 8-gram({n_hidden*2}→{vocab_size})")

# 验证张量形状
with torch.no_grad():
    x_test = Xtr[:4]  # 取4个样本
    y_test = model(x_test)
    print(f"输入: {x_test.shape} → 输出: {y_test.shape}  (期望: torch.Size([4, {vocab_size}]))")

# ==================== 4. 训练 ====================
max_steps = 50000
batch_size = 32
lossi = []  # 记录每个step的loss

print("\n开始训练...")
for i in range(max_steps):
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    # forward
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)

    # backward
    for p in model.parameters():
        p.grad = None
    loss.backward()

    # update — lr 阶梯衰减
    lr = 0.1 if i < 30000 else 0.01
    for p in model.parameters():
        p.data += -lr * p.grad

    if i % 5000 == 0:
        print(f'  step {i:5d}/{max_steps}: loss = {loss.item():.4f}')
    lossi.append(loss.log10().item())

# ==================== 5. 评估 ====================
@torch.no_grad()
def eval_split(m, X, Y, name):
    """计算给定数据集的损失"""
    n = X.shape[0]
    losses = []
    for i in range(0, n, 256):
        x = X[i:i+256]
        y = Y[i:i+256]
        logits = m(x)
        losses.append(F.cross_entropy(logits, y).item())
    avg = sum(losses) / len(losses)
    print(f"  {name}: {avg:.4f}")
    return avg

print("\n评估结果:")
# 切换到 eval 模式（所有 BatchNorm 层的 training=False）
for layer in model.layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = False

eval_split(model, Xtr, Ytr, "Train")
dev_loss = eval_split(model, Xdev, Ydev, "Dev")
eval_split(model, Xte, Yte, "Test")

# 切回训练模式
for layer in model.layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = True

# ==================== 6. 生成/采样 ====================
@torch.no_grad()
def generate(m, n_words=10, max_len=30):
    """从模型中采样名字"""
    for layer in m.layers:
        if isinstance(layer, BatchNorm1D):
            layer.training = False

    results = []
    for _ in range(n_words):
        out = []
        context = [0] * block_size
        while len(out) < max_len:
            x = torch.tensor([context])
            logits = m(x)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1).item()
            if ix == 0:
                break
            context = context[1:] + [ix]
            out.append(ix)
        results.append(''.join(itos[i] for i in out))

    for layer in m.layers:
        if isinstance(layer, BatchNorm1D):
            layer.training = True

    return results

print("\n采样结果:")
names = generate(model, 15)
for i, name in enumerate(names):
    print(f"  {i+1:2d}. {name}")

# ==================== 7. 对比实验：增大模型 ====================
print("\n" + "="*60)
print("实验2: 大模型 (n_emb=48, n_hidden=128)")
print("="*60)

n_emb2, n_hidden2 = 48, 128

model2 = Sequential([
    torch.nn.Embedding(vocab_size, n_emb2),
    FlattenConsecutive(2),
    torch.nn.Linear(n_emb2 * 2, n_hidden2),
    BatchNorm1D(n_hidden2),
    torch.nn.Tanh(),
    FlattenConsecutive(2),
    torch.nn.Linear(n_hidden2 * 2, n_hidden2),
    BatchNorm1D(n_hidden2),
    torch.nn.Tanh(),
    FlattenConsecutive(2),
    torch.nn.Linear(n_hidden2 * 2, vocab_size),
])

n_params2 = sum(p.nelement() for p in model2.parameters())
print(f"参数: {n_params2:,}")

# Kaiming init
with torch.no_grad():
    for layer in model2.layers:
        if isinstance(layer, torch.nn.Linear):
            layer.weight.data *= 5/3 / (layer.weight.shape[1] ** 0.5)
        if isinstance(layer, torch.nn.Embedding):
            layer.weight.data *= 0.1

# 训练
for i in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    Xb, Yb = Xtr[ix], Ytr[ix]

    logits = model2(Xb)
    loss = F.cross_entropy(logits, Yb)

    for p in model2.parameters():
        p.grad = None
    loss.backward()

    lr = 0.1 if i < 30000 else 0.01
    for p in model2.parameters():
        p.data += -lr * p.grad

    if i % 10000 == 0:
        print(f'  step {i:5d}: loss = {loss.item():.4f}')

# 评估
for layer in model2.layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = False

print("\n大模型评估:")
eval_split(model2, Xtr, Ytr, "Train")
dev_loss2 = eval_split(model2, Xdev, Ydev, "Dev")
eval_split(model2, Xte, Yte, "Test")

# 生成
names2 = generate(model2, 15)
print("\n大模型采样:")
for i, name in enumerate(names2):
    print(f"  {i+1:2d}. {name}")

print(f"\n总结: 小模型 Dev={dev_loss:.4f} ({n_params:,}参数) → 大模型 Dev={dev_loss2:.4f} ({n_params2:,}参数)")
