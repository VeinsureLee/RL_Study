import numpy as np
import tqdm
from matplotlib import pyplot as plt

from chapter5_TD.Sarsa import CliffWalkingEnv


# tag:: nstep_Sarsa[]
class nstep_Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_step, num_actions=4):
        self.Q_table = np.zeros((nrow * ncol, num_actions))
        self.n_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_step = n_step
        self.state_list = []
        self.action_list = []
        self.reward_list = []

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state): # Printing optimal actions
        Q_max = np.max(self.Q_table[state])
        best_actions = [0] * self.n_actions
        for i in range(self.n_actions):
            if self.Q_table[state][i] == Q_max:
                best_actions[i] = 1
        return best_actions

    def update(self, state, action, reward, next_state, next_action, done):
        self.state_list.append(state) # <1>
        self.action_list.append(action)
        self.reward_list.append(reward)

        if len(self.state_list) == self.n_step: # <2>
            G = self.Q_table[next_state, next_action]
            for i in reversed(range(self.n_step)):
                G = self.gamma * G + self.reward_list[i]
                if done and i > 0:
                    s = self.state_list[i]
                    a = self.action_list[i]
                    self.Q_table[s,a] += self.alpha * (G - self.Q_table[s,a])
            s = self.state_list.pop(0)
            a = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s,a] += self.alpha * (G - self.Q_table[s,a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []
# end:: nstep_Sarsa[]
# <1> Initialize lists to store states, actions, and rewards
# <2> Check if we have enough steps to perform n-step update

def main():
    np.random.seed(0)
    n_step = 5
    alpha = 0.1
    epsilon = 0.1
    gamma = 0.9
    nrow, ncol = 4, 12
    env = CliffWalkingEnv(nrow, ncol)
    agent = nstep_Sarsa(12, 4, epsilon, alpha, gamma, n_step)
    num_episodes = 500
    return_list = []
    for i in range(10):
        with tqdm.tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                action = agent.take_action(state)
                done = False
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward
                    agent.update(state, action, reward, next_state, next_action, done)
                    state = next_state
                    action = next_action
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每 10 条序列打印一下这 10 条序列的平均回报
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Return')
    plt.title(f'n-step Sarsa (n={n_step}) on Cliff Walking')
    plt.show()

if __name__ == "__main__":
    main()