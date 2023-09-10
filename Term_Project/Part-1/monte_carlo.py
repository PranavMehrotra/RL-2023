# Group/Serial Number: 11 
# Author: Pranav Mehrotra, 20CS10085
# Project Code: TCV1
# Project Title: Controlling a Toy Car around a Grid [Version 1]


import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
import time
import my_env as env
import my_plotter as plotter

GRID_SIZE = 8
PYGAME_CELL_SIZE = 70
GAMMA = 0.9
DIRECTIONS = 4
# MAX_STEPS_IN_ONE_EPISODE = 10000
REWARD_MULTIPLIER = 1



class MonteCarloLearning:
    def __init__(self, grid_size, epsilon, xi, initial_q) -> None:
        self.grid_size = grid_size
        self.directions = DIRECTIONS
        self.num_cells = (grid_size-1) * 4
        self.num_states = self.num_cells * self.directions  # 4 directions per boundary cell
        self.boundaries = [i*(grid_size-1) for i in range(0, grid_size)]
        self.actions = 3  # left, forward, right
        self.epsilon = epsilon
        self.xi = xi

        self.q_table = np.full((self.num_states, self.actions), initial_q)
        self.g_table = np.zeros((self.num_states, self.actions))
        self.count_table = np.zeros((self.num_states, self.actions))
        self.visited = np.zeros((self.num_states, self.actions))
        # self.policy = np.random.randint(0, self.actions, size=self.num_states)
        self.policy = np.zeros(self.num_states, dtype=int)
        self.policy[0] = 1 # start with forward
        self.values = np.zeros(self.num_states)
    

    def get_reward(self, next_cell):
        return 5
    
    def step(self, state, action):
        direction = state % self.directions # 0: up, 1: right, 2: down, 3: left
        cell = state // self.directions
        if(action == 0 or action == 2): # left or right
            direction = (direction + (action - 1) + self.directions) % self.directions
            reward = -1
            next_state = direction + cell * self.directions
            return next_state, reward, False

        # forward
        if(cell in self.boundaries):
            for i in range(len(self.boundaries)):
                if(cell == self.boundaries[i]):
                    if(direction == i+0 or direction == (i+1)%self.directions): # on track
                        next_cell = (cell + 1) if direction == i+0 else (cell - 1 + self.num_cells)%self.num_cells
                        next_state = next_cell * self.directions + direction
                        reward = self.get_reward(next_cell)
                        return next_state, reward, False
                    else: # crossing boundary
                        next_state = state
                        reward = -3
                        return next_state, reward, False
        else:
            temp = cell // (self.grid_size-1)
            if(direction == temp+0 or direction == (temp+2)%self.directions): # on track
                next_cell = (cell + 1)%self.num_cells if direction == temp+0 else (cell - 1)
                next_state = next_cell * self.directions + direction
                reward = self.get_reward(next_cell)
                if(next_cell == 0 and cell == self.num_cells-1):
                    # reward = self.num_cells * REWARD_MULTIPLIER * 3
                    reward = 10
                    return next_state, reward, True
                return next_state, reward, False
            else: # crossing boundary
                next_state = state
                reward = -3
                return next_state, reward, False
            
    def update_g_table(self, state, action, reward, step):
        self.visited[state][action] = step
        self.g_table[ self.visited > 0 ] += ((np.power(np.full(len(self.visited[ self.visited > 0 ]), GAMMA),np.abs(self.visited[ self.visited > 0 ]-step)) * reward))
        # self.g_table[ self.visited > 0 ] += reward
        # For loop over all visited states
        # for i in range(self.num_states):
        #     for j in range(self.actions):
        #         if(self.visited[i][j] > 0):
        #             self.g_table[i][j] += (pow(GAMMA,(step - self.visited[i][j])) * reward)

    def next_action(self, state):
        if(np.random.random() < self.epsilon):
            return np.random.randint(0, self.actions)
        # return self.policy[state]
        return np.argmax(self.q_table, axis=1)[state]
    
    def update_policy(self):
        self.count_table[self.visited > 0] += 1
        self.q_table[self.visited > 0 ] += (self.g_table[self.visited > 0] - self.q_table[self.visited > 0]
                                            ) / self.count_table[self.visited > 0]
        
        self.policy = np.argmax(self.q_table, axis=1)
        self.policy[0] = 1 # start with forward

        self.visited = np.zeros((self.num_states, self.actions))
        self.g_table = np.zeros((self.num_states, self.actions))
        self.epsilon *= self.xi

    def run_one_episode(self):
        state = 0
        done = False
        tot_reward = 0
        steps = 0
        while(not done):
            steps += 1
            action = self.next_action(state)
            next_state, reward, done = self.step(state, action)
            self.update_g_table(state, action, reward, steps)
            tot_reward += reward
            state = next_state
            # if steps >= MAX_STEPS_IN_ONE_EPISODE:
            #     return tot_reward, steps
        self.update_policy()
        return tot_reward, steps
    
    def run(self, episodes=100):
        rewards = np.zeros(episodes)
        steps = np.zeros(episodes)
        for i in range(episodes):
            reward, step = self.run_one_episode()
            rewards[i] = reward
            steps[i] = step
            print(f'Episode {i+1}/{episodes} completed, reward: {reward}, steps: {step}, epsilon: {self.epsilon}', flush=True)

        # Populate optimal state values from Q table
        self.values = np.max(self.q_table, axis=1)

        assert len(self.values) == self.num_states

        return rewards, steps
    
    def average_reward(self, rewards):
        cum_rewards = np.cumsum(rewards)
        avg_rewards = cum_rewards / (np.arange(len(rewards)) + 1)
        return avg_rewards

    def train(self, episodes=100):
        rewards, steps = self.run(episodes)
        
        print("\n------------------------------------------------------------------------------------")
        print("\n\nMonte Carlo training Completed!\nSummary:")
        print(f'# Epochs: {episodes}')
        print(f'average reward: {np.mean(rewards)}')
        # print(f'Steps: {steps}')

        plotter.plot_state_values_heatmap(self.grid_size, self.values, 'monte_carlo')
        plotter.plot_optimal_policy_grid(self.grid_size, self.policy, 'monte_carlo')
        
        # plt.figure(figsize=(10, 5))
        # # plt.subplot(1, 2, 1)
        # plt.plot(self.average_reward(rewards))
        # plt.xlabel("Episodes")
        # plt.ylabel("Average Reward per episode")
        # # plt.subplot(1, 2, 2)
        # # plt.plot(steps)
        # # plt.xlabel("Episodes")
        # # plt.ylabel("Steps")
        # plt.savefig('mclearner_averaged_reward_per_episode.png')


def get_coord(grid_size, cell):
    temp = (grid_size -1)
    if(cell <= temp):
        return 0,temp-cell
    elif(cell <= 2*temp):
        return cell - temp, 0
    elif(cell <= 3*temp):
        return temp, cell - 2*temp
    else:
        return 4*temp - cell,temp

if __name__ == "__main__":
    np.random.seed(42)
    mc = MonteCarloLearning(GRID_SIZE, epsilon=0.7, xi=0.99, initial_q=0.0)
    mc.train(episodes=100)

    # print(f'Final policy: {mc.policy}')

    # Initialize Pygame
    pygame.init()

    # Create environment instance
    env = env.Environment(GRID_SIZE, PYGAME_CELL_SIZE)

    env.reset()
    env.render()
    time.sleep(1) # Delay for visualization
    done = False

    directions_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
    actions_dict = {0: 'left', 1: 'forward', 2: 'right'}

    state = 0
    while(not done):
        env.last_action = actions_dict[mc.policy[state]]
        state, rew , done = mc.step(state, mc.policy[state])
        cell = state // DIRECTIONS
        direct = state % DIRECTIONS
        car_x, car_y = get_coord(grid_size=GRID_SIZE, cell = cell)
        
        env.last_reward = rew
        env.last_done = done
        env.step(car_x, car_y, directions_dict[direct])

    time.sleep(1)
    pygame.quit()

