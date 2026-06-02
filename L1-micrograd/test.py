"""
micrograd 测试 — 三部分：
  Demo 1: 导数基础
  Demo 2: Value 类 + backward 验证
  Demo 3: MLP 训练（四步循环）
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from engine import Value
from nn import MLP

# ============================================================
# Demo 1: 导数基础 — 函数 f(x) 和有限差分
# ============================================================
print("=" * 50)
print("Demo 1: 导数基础")
print("=" * 50)


def f(x):
    return 3 * x ** 2 - 4 * x + 5


print(f"f(3.0) = {f(3.0)}")

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
# plt.plot(xs, ys)
# plt.show()

# 有限差分近似导数: f'(x) ≈ (f(x+h) - f(x)) / h
h = 0.001
x = 3.0
print(f"f'({x}) ≈ { (f(x + h) - f(x)) / h }  (finite difference, h={h})")

# ============================================================
# Demo 2: Value 类 + backward 验证
# ============================================================
print()
print("=" * 50)
print("Demo 2: Value 类 backward 验证")
print("=" * 50)

# d = a*b + c，手动推导梯度：
# ∂d/∂a = b = -3.0
# ∂d/∂b = a = 2.0
# ∂d/∂c = 1.0
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
print(f"d = {d.data}")
print(f"d._prev = {d._prev}")

d.backward()
print()
print("手动推导 vs micrograd:")
print(f"  a.grad: 预期 -3.0,  实际 {a.grad}")
print(f"  b.grad: 预期  2.0,  实际 {b.grad}")
print(f"  c.grad: 预期  1.0,  实际 {c.grad}")

# PyTorch 对比
import torch
x1 = torch.Tensor([2.0]); x1.requires_grad = True
x2 = torch.Tensor([-3.0]); x2.requires_grad = True
x3 = torch.Tensor([10.0]); x3.requires_grad = True
y = x1 * x2 + x3
y.backward()
print()
print("PyTorch 对比:")
print(f"  x1.grad = {x1.grad.item()}")
print(f"  x2.grad = {x2.grad.item()}")
print(f"  x3.grad = {x3.grad.item()}")

# ============================================================
# Demo 3: MLP 训练 — 四步循环
# ============================================================
print()
print("=" * 50)
print("Demo 3: MLP 训练")
print("=" * 50)

# 玩具数据：三个点，拟合目标 y
xs = [[2.0], [3.0], [-1.0]]
ys = [1.0, -1.0, 1.0]

model = MLP(1, [4, 4, 1])  # 1输入 → 4 → 4 → 1输出
lr = 0.02

print(f"模型参数量: {len(model.parameters())}")
print("训练中...")

for epoch in range(500):
    # ① forward
    ypred = [model(x) for x in xs]
    # ② loss
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(ypred, ys))
    # ③ backward
    model.zero_grad()
    loss.backward()
    # ④ update
    for p in model.parameters():
        p.data -= lr * p.grad

    if epoch % 50 == 0:
        print(f"  epoch {epoch:3d}: loss = {loss.data:.4f}")

# 最终预测
print()
print("最终预测 vs 目标:")
for x_in, y_tgt in zip(xs, ys):
    y_out = model(x_in)
    print(f"  x = {x_in[0]:3.0f}  →  y_pred = {y_out.data:7.4f}  (目标: {y_tgt})")
