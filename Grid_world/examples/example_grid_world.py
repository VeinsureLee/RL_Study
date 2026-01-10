import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np
import time

# Example usage:
if __name__ == "__main__":             
    env = GridWorld()
    state = env.reset()
    start = time.time()
    for t in range(1000):
        # env.render()
        action = random.choice(env.action_space)
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")
        if done:
            break
    end = time.time()
    print(f"Time taken: {end-start}")

    # Add policy
    policy_matrix=np.random.rand(env.num_states,len(env.action_space))                                            
    policy_matrix /= policy_matrix.sum(axis=1)[:, np.newaxis]  # make the sum of elements in each row to be 1
    env.render()
    env.add_policy(policy_matrix)

    
    # Add state values
    values = np.random.uniform(0,10,(env.num_states,))
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2)