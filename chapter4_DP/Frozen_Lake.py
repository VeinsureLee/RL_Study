import gymnasium as gym

from chapter4_DP.PI import PolicyIteration, print_agent
from chapter4_DP.VI import ValueIteration

env = gym.make("FrozenLake-v1", render_mode="ansi")
state, info = env.reset()

env = env.unwrapped  # 解封装才能访问状态转移矩阵 P
print(env.unwrapped.desc)

env.render()


holes = set()
ends = set()

# env.P : {state: {action: [(prob, next_state, reward, done), ...]}}
for s in env.P:
    for a in env.P[s]:
        for prob, s_, reward, done in env.P[s][a]:
            if reward == 1.0:  # 目标
                ends.add(s_)
            if done and reward == 0.0:  # done 且没奖励 → 冰洞
                holes.add(s_)

holes = holes - ends  # 去掉目标
print("\n冰洞索引:", holes)
print("目标索引:", ends, "\n")

# 查看目标左边一格（原代码用14，但在4×4 FrozenLake 目标格是15）
target = list(ends)[0]
left_state = target - 1  # 左侧
print(f"状态 {left_state} 的转移情况:")
for a in env.P[left_state]:
    print(env.P[left_state][a])

# 这个动作意义是 Gym 库针对冰湖环境事先规定好的
action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])

action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])