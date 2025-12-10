# tag::ValueIteration[]
class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.n_col * self.env.n_row
        self.pi = [None for i in range(self.env.n_col * self.env.n_row)]

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0 for _ in range(self.env.n_col * self.env.n_row)]
            for s in range(self.env.n_row * self.env.n_col):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_s, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("Value Iteration iter cnt:", cnt)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.n_row * self.env.n_col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_s, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
                qsa_list.append(qsa)
            max_qsa = max(qsa_list)
            max_count = qsa_list.count(max_qsa)
            self.pi[s] = [1.0 / max_count if qsa == max_qsa else 0.0 for qsa in qsa_list]
        print("Policy derived from Value Iteration.")
        return self.pi

# end:: ValueIteration[]

def main():
    from chapter4_DP.Clif_Walking import CliffWalkingEnv
    from chapter4_DP.PI import print_agent
    env = CliffWalkingEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])

if __name__ == "__main__":
    main()