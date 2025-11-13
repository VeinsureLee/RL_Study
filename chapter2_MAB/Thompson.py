from epsilon_greedy import bandit_10_arm,Solver,plot_results
import numpy as np

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super().__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为 1 的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为 0 的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照 Beta 分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新 Beta 分布的第一个参数
        self._b[k] += (1 - r)  # 更新 Beta 分布的第二个参数
        return k

def main():
    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(5000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results([thompson_sampling_solver], ["ThompsonSampling"])
if __name__ == '__main__':
    main()