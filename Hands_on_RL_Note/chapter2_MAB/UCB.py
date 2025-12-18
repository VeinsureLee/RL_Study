from epsilon_greedy import Solver,bandit_10_arm,plot_results
import numpy as np


class UCB(Solver):
    def __init__(self, bandit, coef, initial_prob=1.0):
        super().__init__(bandit)
        self.coef = coef
        self.estimates = np.array([initial_prob] * self.bandit.K)
        self.total_counts = 0
    def run_one_step(self):
        self.total_counts += 1
        ucb_values = self.estimates + self.coef * np.sqrt(np.log(self.total_counts) / (2*(self.counts + 1e-5)))
        k = np.argmax(ucb_values)
        reward = self.bandit.step(k)
        # update estimate
        n = self.counts[k]
        self.estimates[k] = (n * self.estimates[k] + reward) / (n + 1)
        return k


def main():
    np.random.seed(1)
    coef = 1 # 控制不确定性比重的系数
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(5000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results([UCB_solver], ["UCB"])

if __name__ == '__main__':
    main()