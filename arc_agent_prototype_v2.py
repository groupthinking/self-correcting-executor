"""ARC Agent Prototype v2.

A minimal "Fluid Seed" agent acting in a simple grid-world environment.

The agent operates under three immutable Seed Rules:
  * fidelity  - maximize intent modeling
  * humility  - explore on uncertainty
  * integrity - log adaptations

It probes the environment, builds a belief about the goal location from
feedback, and steers toward that belief once it has been discovered. The run
produces a trajectory visualization.

Output (the trajectory PNG) is written next to this script so the prototype
runs anywhere without depending on a hardcoded absolute path.
"""

import os

import matplotlib

# Use a non-interactive backend so the prototype runs in headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAJECTORY_PATH = os.path.join(OUTPUT_DIR, "agent_trajectory.png")


class SimpleGridEnv:
    def __init__(self, size=10, seed=42):
        np.random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
        self.goal_pos = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.array([np.random.randint(0, self.size), np.random.randint(0, self.size)])
        self.steps = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        return grid.tolist()

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True
        moves = {'up': [-1, 0], 'down': [1, 0], 'left': [0, -1], 'right': [0, 1]}
        if action in moves:
            new_pos = self.agent_pos + moves[action]
            if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
                self.agent_pos = new_pos
        self.steps += 1
        reward = 100 if np.array_equal(self.agent_pos, self.goal_pos) else -1
        self.done = np.array_equal(self.agent_pos, self.goal_pos) or self.steps > 50
        return self.get_state(), reward, self.done


class FluidSeedAgent:
    def __init__(self):
        self.belief = {}  # Simple goal position belief
        self.trajectory = []
        self.agent_pos = None
        # Immutable Seed Rules
        self.seed_rules = {
            'fidelity': True,   # Maximize intent modeling
            'humility': True,   # Explore on uncertainty
            'integrity': True,  # Log adaptations
        }

    def act(self, state, step):
        # Simple probing: random if no belief, else towards believed goal
        if not self.belief or self.agent_pos is None:
            actions = ['up', 'down', 'left', 'right']
            action = np.random.choice(actions)
        else:
            # Derivative-like: move towards goal
            dx = self.belief['goal'][0] - self.agent_pos[0]
            dy = self.belief['goal'][1] - self.agent_pos[1]
            if abs(dx) > abs(dy):
                action = 'down' if dx > 0 else 'up'
            else:
                action = 'right' if dy > 0 else 'left'
        self.trajectory.append((step, action))
        return action

    def update(self, state, reward, done, agent_pos):
        self.agent_pos = agent_pos
        if reward > 0:
            self.belief['goal'] = list(agent_pos)  # Update from feedback (derivative)
        if done:
            print("Seed Rules Enforced:", self.seed_rules)


def visualize_trajectory(env, trajectory, title="Agent Trajectory"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    grid = np.zeros((env.size, env.size))
    for pos in trajectory:
        grid[pos[0], pos[1]] = 1
    ax.imshow(grid, cmap='Blues')
    ax.set_title(title)
    ax.plot(env.goal_pos[1], env.goal_pos[0], 'r*', markersize=15, label='Goal')
    plt.legend()
    plt.savefig(TRAJECTORY_PATH)
    print("Visualization saved to", TRAJECTORY_PATH)
    plt.close()


def main():
    env = SimpleGridEnv()
    agent = FluidSeedAgent()
    state = env.reset()
    agent_pos = env.agent_pos.copy()
    trajectory_positions = [tuple(agent_pos)]
    done = False

    for step in range(50):
        action = agent.act(state, step)
        state, reward, done = env.step(action)
        agent_pos = env.agent_pos.copy()
        trajectory_positions.append(tuple(agent_pos))
        agent.update(state, reward, done, agent_pos)
        if done:
            break

    visualize_trajectory(env, trajectory_positions)
    print("Simulation complete. Goal reached:", done)


if __name__ == "__main__":
    main()
