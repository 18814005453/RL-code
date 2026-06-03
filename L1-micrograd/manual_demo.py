"""
手动正向→反向→更新，不用任何库，不用 Value 类
y = w*x + b,  x=1.5, y_target=0.8,  L = ½(y_pred - y_target)²
"""

x = 1.5
y_target = 0.8
w = 0.8
b = 0.2
lr = 0.1

for epoch in range(10):
    # ① forward
    y_pred = w * x + b
    loss = 0.5 * (y_pred - y_target) ** 2

    # ② backward（手算导数）
    dL_dy = y_pred - y_target   # ∂L/∂y_pred
    dL_dw = dL_dy * x           # ∂L/∂w
    dL_db = dL_dy * 1           # ∂L/∂b

    # ③ update
    w -= lr * dL_dw
    b -= lr * dL_db

    print(f"epoch {epoch:2d}:  y_pred={y_pred:6.4f}  loss={loss:.4f}  w={w:6.4f}  b={b:6.4f}  grad_w={dL_dw:6.4f}  grad_b={dL_db:6.4f}")
