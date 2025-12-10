import copy
# tag::Policy Iteration[]
class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.ncol * self.env.nrow  # <1>
        self.pi = [[0.25,0.25,0.25,0.25] for _ in range(self.env.ncol * self.env.nrow)]  # <2>

        # <1> state value
        # <2> random policy initialization(equal probability for each action)

    # tag:: policy_evaluation[]
    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_v = [0 for _ in range(self.env.ncol * self.env.nrow)]
            for s in range(self.env.nrow * self.env.ncol):
                qsa_list = [] # <1>
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_s, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_s] * (1 - done)) # <2>
                    qsa_list.append(qsa * self.pi[s][a])
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: # <3>
                break
            cnt += 1
        print("Policy Evaluation iter cnt:", cnt)
        # <1> Begin to calculate every q(s,a) at state s
        # <2> Reward r is related to next state
        # <3> Stop condition: convergence close enough to real value
    # end:: policy_evaluation[]

    # tag:: policy_improvement[]
    def policy_improvement(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_s, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
                qsa_list.append(qsa)
            max_qsa = max(qsa_list)
            max_count = qsa_list.count(max_qsa) # <1>
            self.pi[s] = [1.0 / max_count if qsa == max_qsa else 0.0 for qsa in qsa_list] # <2>
        print("Policy Improvement done.")
        return self.pi
        # <1> Count how many actions have the same max value
        # <2> Update policy to be greedy
    # end:: policy_improvement[]

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi) # <1>
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break
        # <1> Deep copy to compare old and new policy

# end:: Policy Iteration[]

# tag:: print_agent[]
def print_agent(agent, action_meaning, disaster=None, end=None):
    if end is None:
        end = []
    if disaster is None:
        disaster = []
    print("State Value: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()
    print("Policy: ")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            s = i * agent.env.ncol + j
            if s in disaster:
                print('****', end=' ')
            elif s in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[s]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()
# end:: print_agent[]

def main():
    from chapter4_DP.Clif_Walking import CliffWalkingEnv
    env = CliffWalkingEnv()
    theta = 1e-3
    gamma = 0.9
    pi_agent = PolicyIteration(env, theta, gamma)
    pi_agent.policy_iteration()
    action_meaning = ['^', 'v', '<', '>']
    print_agent(pi_agent, action_meaning, disaster=list(range(37, 47)), end=[47])

if __name__ == "__main__":
    main()