import numpy as np
from collections import defaultdict
import random


class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon  # 用于 ε-贪心策略
        self.V = defaultdict(float)  # 状态值函数
        self.returns = defaultdict(list)  # 用于存储每个状态的回报列表
        self.policy = self._init_policy()

    # 初始化随机策略
    def _init_policy(self):
        policy = {}
        for x in range(self.env.env_size[0]):
            for y in range(self.env.env_size[1]):
                state = (x, y)
                if state in self.env.forbidden_states or state == self.env.target_state:
                    continue
                policy[state] = random.choice(self.env.action_space)
        return policy

    # ε-贪心选择动作
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.action_space)
        else:
            return self.policy.get(state, (0, 0))

    # 蒙特卡洛策略评估
    def evaluate_policy(self, episodes=500):
        for ep in range(episodes):
            state, _ = self.env.reset()
            trajectory = []  # 存储 (state, reward)
            done = False

            # 生成一条完整轨迹
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                trajectory.append((state, reward))
                state = next_state

            # 计算回报并更新状态值
            G = 0
            for state, reward in reversed(trajectory):
                G = self.gamma * G + reward
                # 只对轨迹中首次出现的状态进行更新（first-visit MC）
                if state not in [s for s, _ in trajectory[:-1]]:
                    self.returns[state].append(G)
                    self.V[state] = np.mean(self.returns[state])

        return self.V

    # 蒙特卡洛策略改进（可选）
    def improve_policy(self):
        for state in self.policy:
            best_action = None
            best_value = -float('inf')
            for action in self.env.action_space:
                next_state, reward = self.env._get_next_state_and_reward(state, action)
                value = reward + self.gamma * self.V.get(next_state, 0)
                if value > best_value:
                    best_value = value
                    best_action = action
            self.policy[state] = best_action


import sys

sys.argv = ['']
sys.path.append("../..")
from examples.arguments import args
from src.grid_world import GridWorld


env = GridWorld()
agent = MonteCarloAgent(env, gamma=0.9, epsilon=0.1)

# 蒙特卡洛策略评估
V = agent.evaluate_policy(episodes=10)
print("State values:")
for y in range(env.env_size[1]):
    for x in range(env.env_size[0]):
        print(round(V.get((x, y), 0), 1), end="\t")
    print()

# 可视化状态值
env.render_static(values=[V.get((x, y), 0) for y in range(env.env_size[1]) for x in range(env.env_size[0])])
