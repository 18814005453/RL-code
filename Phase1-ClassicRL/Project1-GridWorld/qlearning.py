"""
Project 1: Grid World Q-Learning
纯 Python，不用任何 ML 库。智能体在 5×5 格子中从起点走到终点，避开陷阱。
"""
import random

# ==================== 1. 环境：5×5 格子世界 ====================
SIZE = 5
GOAL = (4, 4)
TRAPS = [(1, 2), (2, 3), (3, 1)]
START = (0, 0)

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上 下 左 右

def step(state, action):
    r, c = state
    dr, dc = action
    nr, nc = r + dr, c + dc
    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        return state, -1, False  # 撞墙
    if (nr, nc) in TRAPS:
        return (nr, nc), -1, True
    if (nr, nc) == GOAL:
        return (nr, nc), 1, True
    return (nr, nc), 0, False

# ==================== 2. Q-table + 训练 ====================
Q = {}
for r in range(SIZE):
    for c in range(SIZE):
        Q[(r, c)] = [0.0, 0.0, 0.0, 0.0]

alpha = 0.1     # 学习率
gamma = 0.9     # 折扣因子
epsilon = 1.0   # 探索率
episodes = 500

for ep in range(episodes):
    state = START
    while True:
        # ε-greedy 选择动作
        if random.random() < epsilon:
            a = random.randint(0, 3)
        else:
            a = max(range(4), key=lambda i: Q[state][i])

        next_state, reward, done = step(state, ACTIONS[a])

        # Q-learning 更新
        next_max = max(Q[next_state])
        Q[state][a] += alpha * (reward + gamma * next_max - Q[state][a])

        if done:
            break
        state = next_state

    epsilon = max(0.01, epsilon * 0.995)  # 逐渐减少探索

# ==================== 3. 展示结果 ====================
print("训练完成\n")
print("最优路径（贪心策略）：")
state = START
path = [state]
while state != GOAL:
    a = max(range(4), key=lambda i: Q[state][i])
    next_state, _, done = step(state, ACTIONS[a])
    if next_state == state:  # 撞墙了，跳过
        qs = Q[state][:]
        qs[a] = -float('inf')
        a = max(range(4), key=lambda i: qs[i])
        next_state, _, _ = step(state, ACTIONS[a])
    path.append(next_state)
    state = next_state
    if len(path) > 50:
        print("  路径过长，停止")
        break

print(f"  {' → '.join(str(p) for p in path)}")
print(f"  步数: {len(path)-1}")

print("\n格子地图：")
for r in range(SIZE):
    row = []
    for c in range(SIZE):
        if (r, c) == START:     row.append("S ")
        elif (r, c) == GOAL:     row.append("G ")
        elif (r, c) in TRAPS:    row.append("X ")
        elif (r, c) in path:     row.append("· ")
        else:                    row.append(". ")
    print("  " + "".join(row))

print("\nQ-table 概览（每个格子的最优动作方向）：")
ARROWS = ["↑", "↓", "←", "→"]
for r in range(SIZE):
    row = []
    for c in range(SIZE):
        if (r, c) == GOAL or (r, c) in TRAPS:
            row.append("· ")
        else:
            a = max(range(4), key=lambda i: Q[(r, c)][i])
            row.append(f"{ARROWS[a]} ")
    print("  " + "".join(row))
