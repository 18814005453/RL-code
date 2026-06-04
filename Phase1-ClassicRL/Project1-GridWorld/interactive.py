"""
交互式 Grid World — 你加陷阱，智能体实时找路
命令：add x y（加陷阱）| remove x y（移除）| go（重训+显示路径）| reset（清空陷阱）| quit
"""
import random

SIZE = 5
START = (0, 0)
GOAL = (SIZE-1, SIZE-1)
traps = set()

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ARROWS = ["↑", "↓", "←", "→"]

def step(state, action, traps):
    r, c = state
    dr, dc = action
    nr, nc = r + dr, c + dc
    if not (0 <= nr < SIZE and 0 <= nc < SIZE):
        return state, -1, False
    if (nr, nc) in traps:
        return (nr, nc), -1, True
    if (nr, nc) == GOAL:
        return (nr, nc), 1, True
    return (nr, nc), 0, False

def train(traps, episodes=2000):
    Q = {}
    for r in range(SIZE):
        for c in range(SIZE):
            Q[(r, c)] = [0.0, 0.0, 0.0, 0.0]
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    for _ in range(episodes):
        state = START
        while True:
            if random.random() < epsilon:
                a = random.randint(0, 3)
            else:
                a = max(range(4), key=lambda i: Q[state][i])
            next_state, reward, done = step(state, ACTIONS[a], traps)
            next_max = max(Q[next_state])
            Q[state][a] += alpha * (reward + gamma * next_max - Q[state][a])
            if done:
                break
            state = next_state
        epsilon = max(0.01, epsilon * 0.998)
    return Q

def find_path(Q, traps):
    state = START
    path = [state]
    for _ in range(200):
        if state == GOAL:
            break
        a = max(range(4), key=lambda i: Q[state][i])
        next_state, _, _ = step(state, ACTIONS[a], traps)
        path.append(next_state)
        state = next_state
    return path

def show_grid(path, traps, Q=None):
    print()
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            if (r, c) == START:
                row.append("\033[42mS \033[0m")
            elif (r, c) == GOAL:
                row.append("\033[42mG \033[0m")
            elif (r, c) in traps:
                row.append("\033[41mX \033[0m")
            elif (r, c) in path:
                row.append("\033[43m· \033[0m")
            else:
                row.append(". ")
        print("  " + "".join(row))

    # Q 表方向
    if Q:
        print()
        for r in range(SIZE):
            dirs = []
            for c in range(SIZE):
                if (r, c) == GOAL or (r, c) in traps:
                    dirs.append("· ")
                else:
                    a = max(range(4), key=lambda i: Q[(r, c)][i])
                    dirs.append(f"{ARROWS[a]} ")
            print("  " + "".join(dirs))

def main():
    traps = set([(1, 2), (2, 3), (3, 1)])  # 初始陷阱（同 P1）
    print("\033[2J\033[H", end="")  # 清屏
    print("=" * 50)
    print("  交互式 Grid World — 你加陷阱，智能体实时找路")
    print("=" * 50)
    print("  add x y   — 在 (x,y) 加一个陷阱")
    print("  remove x y — 移除 (x,y) 的陷阱")
    print("  go        — 重训并显示路径")
    print("  reset     — 清空所有陷阱")
    print("  quit      — 退出")
    print("=" * 50)

    # 初始训练
    Q = train(traps)
    path = find_path(Q, traps)
    print(f"\n初始状态  |  陷阱: {sorted(traps)}")
    show_grid(path, traps, Q)
    path_len = len(path) - 1
    print(f"  步数: {path_len}  {'✅ 可达' if path[-1] == GOAL else '❌ 找不到路!'}")

    while True:
        try:
            cmd = input("\n> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not cmd:
            continue

        if cmd[0] == "quit" or cmd[0] == "q":
            break

        elif cmd[0] == "add" and len(cmd) == 3:
            x, y = int(cmd[1]), int(cmd[2])
            if (x, y) in [START, GOAL]:
                print("  不能放在起点或终点！")
                continue
            traps.add((x, y))
            print(f"  陷阱 +({x},{y})  |  共 {len(traps)} 个陷阱")

        elif cmd[0] == "remove" and len(cmd) == 3:
            x, y = int(cmd[1]), int(cmd[2])
            traps.discard((x, y))
            print(f"  陷阱 -({x},{y})  |  共 {len(traps)} 个陷阱")

        elif cmd[0] == "reset":
            traps.clear()
            print("  全部陷阱已清除")

        elif cmd[0] == "go":
            print("  重训中...", end="", flush=True)
            Q = train(traps)
            path = find_path(Q, traps)
            show_grid(path, traps, Q)
            path_len = len(path) - 1
            if path[-1] == GOAL:
                print(f"  步数: {path_len}  ✅")
            else:
                print(f"  步数: {path_len}  ❌ 终点不可达——智能体被困住了")

        else:
            print("  不认识。试试：add 2 3 / remove 2 3 / go / reset / quit")


if __name__ == "__main__":
    main()
