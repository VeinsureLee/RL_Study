import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# tag:: CliffWalkingEnv[]
class CliffWalkingEnv:
    def __init__(self, nrow, ncol):
        self.nrow = nrow  # number of rows in the grid
        self.ncol = ncol  # number of columns in the grid
        self.x = 0  # row coordinate (vertical position)
        self.y = 0  # column coordinate (horizontal position), initial position corrected to (0,0) (bottom-left corner)

    def step(self, action):  # <1>
        # Define action effects: 0-up, 1-down, 2-left, 3-right
        dx, dy = [[-1, 0], [1, 0], [0, -1], [0, 1]][action]  # <2>

        # Boundary restriction: ensure coordinates do not cross boundaries
        self.x = np.clip(self.x + dx, 0, self.nrow - 1)
        self.y = np.clip(self.y + dy, 0, self.ncol - 1)

        reward = -1
        done = False

        # Correct: state encoding = row index * number of columns + column index
        current_state = self.x * self.ncol + self.y

        # Termination condition: reach the target position (bottom-right corner (nrow-1, ncol-1))
        if self.x == self.nrow - 1 and self.y == self.ncol - 1:
            done = True
        # Cliff area: last row (except start and end points)
        elif self.x == self.nrow - 1 and 0 < self.y < self.ncol - 1:
            reward = -100  # penalty for falling into the cliff
            done = True  # end episode immediately when falling into cliff

        return current_state, reward, done

    def reset(self):
        # Reset to start point: bottom-left corner (nrow-1, 0)
        self.x = self.nrow - 1
        self.y = 0
        return self.x * self.ncol + self.y


# end:: CliffWalkingEnv[]
# <1>State encoding fixed: row*col + column index to match grid state logic
# <2>Action definition optimized for readability: up/down/left/right

# tag:: Sarsa[]
class Sarsa:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, num_actions=4):
        self.Q_table = np.zeros((nrow * ncol, num_actions))  # number of states = rows*columns
        self.num_actions = num_actions  # 4 actions: up, down, left, right
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha  # learning rate

    def take_action(self, state):
        # epsilon-greedy policy to select action
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):
        # Get optimal actions for current state (may be multiple)
        Q_max = np.max(self.Q_table[state])
        best_actions = [0] * self.num_actions
        for i in range(self.num_actions):
            if self.Q_table[state][i] == Q_max:
                best_actions[i] = 1
        return best_actions

    def update(self, state, action, reward, next_state, next_action):
        # Sarsa update formula: TD(0)
        td_error = reward + self.gamma * self.Q_table[next_state][next_action] - self.Q_table[state][action]
        self.Q_table[state][action] += self.alpha * td_error


# end:: Sarsa[]

def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    """
    Print policy grid
    :param agent: the reinforcement learning agent
    :param env: cliff walking environment
    :param action_meaning: action symbols ['^','v','<','>']
    :param disaster: state indices of dangerous area (cliff)
    :param end: state index of end point
    """
    for i in range(env.nrow):  # iterate over each row
        for j in range(env.ncol):  # iterate over each column
            state = i * env.ncol + j  # calculate state index of current grid
            if state in disaster:
                print('****', end=' ')  # cliff area
            elif state in end:
                print('EEEE', end=' ')  # end point
            else:
                # Get optimal actions and format output
                best_actions = agent.best_action(state)
                pi_str = ''.join([action_meaning[k] if best_actions[k] else 'o' for k in range(len(action_meaning))])
                print(pi_str, end=' ')
        print()  # new line


def main():
    # Environment parameters: 4 rows, 12 columns (classic cliff walking configuration)
    nrow, ncol = 4, 12
    env = CliffWalkingEnv(nrow, ncol)

    # Random seed (ensure reproducibility)
    np.random.seed(0)

    # Sarsa hyperparameters
    epsilon = 0.1  # exploration rate
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
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
                    episode_return += reward
                    agent.update(state, action, reward, next_state, next_action)
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


if __name__ == '__main__':
    main()