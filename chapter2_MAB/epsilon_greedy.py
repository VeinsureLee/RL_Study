import matplotlib.pyplot as plt
import numpy as np


class BernoulliBandit:
    # K is the number of arms
    def __init__(self, K):
        self.K = K
        self.probs=np.random.uniform(size=self.K)
        self.best_idx=np.argmax(self.probs)
        self.best_prob=self.probs[self.best_idx]

    def step(self,k):
        if np.random.rand()<self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)
K=10
bandit_10_arm=BernoulliBandit(K)
print("The number of arms:",bandit_10_arm.K)
print("The best arm index:",bandit_10_arm.best_idx)
print("The best arm probability:",bandit_10_arm.best_prob)


class Solver:
    def __init__(self,bandit):
        self.bandit=bandit
        self.counts=np.zeros(bandit.K) # number of times each arm is played
        self.regret=0
        self.actions=[]
        self.regrets=[]
    def update_regret(self,k):
        self.regret+=self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)
    def run_one_step(self):
        raise NotImplementedError
    def run(self,num_steps):
        for _ in range(num_steps):
            k=self.run_one_step()
            self.counts[k]+=1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self,bandit,epsilon=0.01,initial_prob=1.0):
        super().__init__(bandit)
        self.epsilon=epsilon
        self.estimates=np.array([initial_prob]*self.bandit.K)
    def run_one_step(self):
        if np.random.rand()<self.epsilon:
            # explore
            k=np.random.randint(0,self.bandit.K)
        else:
            # exploit
            k=np.argmax(self.estimates)
        reward=self.bandit.step(k)
        # update estimate
        n=self.counts[k]
        self.estimates[k]=(self.estimates[k]*n + reward)/(n+1)
        return k


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


# Example usage
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm)
epsilon_greedy_solver.run(5000)
print('epsilon-greedy regret:', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


# Compare different epsilon values
np.random.seed(1)
epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


class DecayingEpsilonGreedy(Solver):
    def __init__(self,bandit,initial_epsilon=1.0,decay_rate=0.99,initial_prob=1.0):
        super().__init__(bandit)
        self.initial_epsilon=initial_epsilon
        self.decay_rate=decay_rate
        self.estimates=np.array([initial_prob]*self.bandit.K)
    def run_one_step(self):
        t=len(self.actions)
        epsilon=self.initial_epsilon*(self.decay_rate**t)
        if np.random.rand()<epsilon:
            # explore
            k=np.random.randint(0,self.bandit.K)
        else:
            # exploit
            k=np.argmax(self.estimates)
        reward=self.bandit.step(k)
        # update estimate
        n=self.counts[k]
        self.estimates[k]=(self.estimates[k]*n + reward)/(n+1)
        return k


np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print('epsilon 值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
print('a')
print('b')
print('e')
print('c')
print('d')