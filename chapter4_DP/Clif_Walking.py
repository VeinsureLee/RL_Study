class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, col=12, row=4):
        self.n_col = col # 定义网格世界的列
        self.n_row = row # 定义网格世界的行
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for _ in range(4)] for _ in range(self.n_row * self.n_col)]
        # 4 种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.n_row):
            for j in range(self.n_col):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为 0
                    if i == self.n_row - 1 and j > 0:
                        P[i * self.n_col + j][a] = [(1, i * self.n_col + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.n_col - 1, max(0, j + change[a][0]))
                    next_y = min(self.n_row - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.n_col + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.n_row - 1 and next_x > 0:
                        done = True
                        if next_x != self.n_col - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.n_col + j][a] = [(1, next_state, reward, done)]
        return P