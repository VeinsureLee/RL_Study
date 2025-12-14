import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from chapter5_TD.Sarsa import CliffWalkingEnv, print_agent

# tag:: Expectation_Sarsa[]
class Expectation_Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, num_actions=4):
        self.Q_table = np.zeros((nrow * ncol, num_actions))
        self.n_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):  # Printing optimal actions
        Q_max = np.max(self.Q_table[state])
        best_actions = [0] * self.n_actions
        for i in range(self.n_actions):
            if self.Q_table[state][i] == Q_max:
                best_actions[i] = 1
        return best_actions

    def update(self, state, action, reward, next_state, done):
        # Calculate expected value of next state
        policy_s = np.ones(self.n_actions) * self.epsilon / self.n_actions
        q_next = self.Q_table[next_state]
        best_actions = np.where(q_next == np.max(q_next))[0]
        policy_s[best_actions] += (1 - self.epsilon) / len(best_actions)
        expected_Q_next = np.dot(q_next, policy_s)

        target = reward + (0 if done else self.gamma * expected_Q_next)
        self.Q_table[state, action] += self.alpha * (target - self.Q_table[state, action])

def main():
    # Environment parameters: 4 rows, 12 columns (classic cliff walking configuration)
    nrow, ncol = 4, 12
    env = CliffWalkingEnv(nrow, ncol)

    # Random seed (ensure reproducibility)
    np.random.seed(0)

    # Sarsa hyperparameters
    epsilon = 0.1  # exploration rate
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor <1>
    agent = Expectation_Sarsa(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # total training episodes

    # Record return for each episode
    return_list = []
    for i in range(10):  # display progress in 10 stages
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # corrected variable name for semantic clarity
                state = env.reset()
                action = agent.take_action(state)
                done = False

                # Single episode interaction
                while not done:
                    next_state, reward, done = env.step(action)
                    next_action = agent.take_action(next_state)
                    episode_return += reward # <2>
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    action = next_action

                return_list.append(episode_return)

                # Update progress bar every 10 episodes
                if (i_episode + 1) % 10 == 0:
                    avg_return = np.mean(return_list[-10:])
                    pbar.set_postfix({
                        'episode': f'{num_episodes / 10 * i + i_episode + 1}',
                        'avg_return': f'{avg_return:.3f}'
                    })
                pbar.update(1)

    # Plot return curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(return_list)), return_list, label='Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Return per Episode')
    plt.title('Sarsa on Cliff Walking')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # Print final policy
    action_meaning = ['^', 'v', '<', '>']  # action symbols
    cliff_states = [36 + j for j in range(1, 11)]  # cliff area: row 3 (index 3), columns 1-10
    end_state = [3 * 12 + 11]  # end point: row 3, column 11 (index 47)
    print('\nSarsa 算法最终收敛的策略为：')
    print_agent(agent, env, action_meaning, cliff_states, end_state)

if __name__ == "__main__":
    main()