"""
γ 对比实验 — 15格走廊。起点在左，终点在右，看 gamma 怎么影响奖励信号往回传多远。
"""
import random

LENGTH = 15
START = 0
GOAL = 14
ACTIONS = [-1, 1]

def step(state, action):
    nxt = state + action
    if nxt < 0:    return 0, -1, False
    if nxt >= LENGTH: return LENGTH - 1, -1, False
    if nxt == GOAL: return nxt, 1, True
    return nxt, 0, False

def train(gamma, episodes):
    Q = {s: [0.0, 0.0] for s in range(LENGTH)}
    alpha = 0.1
    epsilon = 1.0
    for ep in range(episodes):
        state = START
        while True:
            if random.random() < epsilon:
                a = random.randint(0, 1)
            else:
                a = max([0, 1], key=lambda i: Q[state][i])
            next_state, reward, done = step(state, ACTIONS[a])
            next_max = max(Q[next_state])
            Q[state][a] += alpha * (reward + gamma * next_max - Q[state][a])
            if done:
                break
            state = next_state
        epsilon = max(0.01, epsilon * 0.995)
    return Q

print("15 格走廊，start=0, goal=14。每格只看 Q(→) 的值（往右走的值）")
print("Q 值越大说明奖励信号已经传到这一格\n")

for ep in [100, 400, 1000]:
    print(f"--- episodes={ep} ---")
    Q09 = train(0.9, ep)
    Q01 = train(0.1, ep)
    print(f"{'格':>3}: ", end="")
    for s in range(LENGTH):
        print(f"{s:>5}", end="")
    print()
    print(f"γ=0.9: ", end="")
    for s in range(LENGTH):
        print(f"{Q09[s][1]:>5.2f}", end="")
    print()
    print(f"γ=0.1: ", end="")
    for s in range(LENGTH):
        print(f"{Q01[s][1]:>5.2f}", end="")
    print("\n")
