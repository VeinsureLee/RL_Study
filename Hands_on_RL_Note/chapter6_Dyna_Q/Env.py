import numpy as np


class CliffWalkingEnv:
    def __init__(self, nrow, ncol):
        self.nrow = nrow  # number of rows in the grid
        self.ncol = ncol  # number of columns in the grid
        self.x = nrow - 1    # row coordinate (vertical position)
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