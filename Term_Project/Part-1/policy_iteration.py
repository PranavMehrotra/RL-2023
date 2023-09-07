import numpy as np
import matplotlib.pyplot as plt
import pygame
import math
import time
import env


GRID_SIZE = 7
GAMMA = 0.9
DIRECTIONS = 4
REWARD_MULTIPLIER = 1



class PolicyIteration:
    def __init__(self, grid_size, gamma, init_policy=None) -> None:
        self.grid_size = grid_size
        self.gamma = gamma
        self.directions = DIRECTIONS # up, down, left, right
        self.num_cells = (grid_size-1) * 4
        self.num_states = self.num_cells * self.directions  # 4 directions per boundary cell
        self.boundaries = [i*(grid_size-1) for i in range(0, grid_size)]
        print(f'num_cells: {self.num_cells}, num_states: {self.num_states}, boundaries: {self.boundaries}')
        self.actions = 3  # left, forward, right
        self.values = np.zeros(self.num_states)
        if init_policy is None:
            self.policy = np.random.randint(0, self.actions, self.num_states)
            self.policy[0] = 1 # start with forward
        else:
            if(len(init_policy)!=self.num_states):
                raise Exception("The length of init_policy must be equal to the number of states.")
            self.policy = init_policy
    
    def get_reward(self, next_cell):
        # if(next_cell in self.boundaries):
        #     return next_cell * REWARD_MULTIPLIER * 2
        # return next_cell * REWARD_MULTIPLIER
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
                        reward = -1
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
                reward = -1
                return next_state, reward, False


    def policy_eval_one_step(self):
        change = 0
        for state in range(self.num_states):
            temp = self.values[state]
            action = self.policy[state]
            next_state, reward, done = self.step(state, action)
            self.values[state] = reward + self.gamma * self.values[next_state]
            change = max(change, abs(temp - self.values[state]))
        return change
    
    def policy_eval(self):
        convergence=1e-3
        iters = 0
        change = self.policy_eval_one_step()
        while iters < 500:
            iters += 1
            change = self.policy_eval_one_step()
            if change < convergence:
                print(f'policy evaluation converged at {iters} iterations')
                break
        return iters
    
    def policy_improvement(self):
        policy_improve_count = 0
        for state in range(1,self.num_states): # start with 1 to skip start state(no update needed)
            temp = self.policy[state]
            action_reward_list = np.zeros(self.actions)
            for action in range(self.actions):
                next_state, reward, done = self.step(state, action)
                action_reward_list[action] = reward + self.gamma * self.values[next_state]
            self.policy[state] = np.argmax(action_reward_list)
            if temp != self.policy[state]:
                policy_improve_count += 1
        print(f'policy improvement changed {policy_improve_count} actions')
        return policy_improve_count
    
    def policy_iteration(self):
        iters = 0
        print(f'initial policy: {self.policy}')
        print("start policy iteration")
        eval_iters = self.policy_eval()
        eval_iters_list = [eval_iters]
        policy_improve_count = self.policy_improvement()
        policy_improve_count_list = [policy_improve_count]
        while iters < 100:
            iters += 1
            print("----------------------------------------------------------------------------------")
            print(f'iteration {iters}')
            new_eval_iters = self.policy_eval()
            new_policy_improve_count = self.policy_improvement()
            eval_iters_list.append(new_eval_iters)
            policy_improve_count_list.append(new_policy_improve_count)
            if new_policy_improve_count == 0:
                break

        return eval_iters_list, policy_improve_count_list
    

    def train(self, plot=True):
        eval_iters_list, policy_improve_count_list = self.policy_iteration()
        
        print(f'# epochs: {len(policy_improve_count_list)}')
        print(f'eval count = {eval_iters_list}')
        print(f'policy change = {policy_improve_count_list}')

        if plot:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(eval_iters_list)
            plt.xlabel("Iterations")
            plt.ylabel("Number of policy evaluation steps")
            plt.subplot(1, 2, 2)
            plt.plot(policy_improve_count_list)
            plt.xlabel("Iterations")
            plt.ylabel("Number of policy improvement steps")
            plt.show()


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
    grid_size = GRID_SIZE
    gamma = GAMMA
    np.random.seed(42)
    policy_iteration = PolicyIteration(grid_size, gamma)
    policy_iteration.train(False)

    print(f'final policy: {policy_iteration.policy}')

    # Initialize Pygame
    pygame.init()

    # Create environment instance
    env = env.Environment(grid_size)

    # Example of using the environment
    env.reset()
    env.render()
    time.sleep(1) # Delay for visualization
    done = False
    # for _ in range(2):
    #     action = 'forward'
    #     next_state, reward, done, _ = env.step(action)
    #     env.render()
    
    directions_dict = {
        0 : 'up',
        1 : 'right',
        2 : 'down',
        3 : 'left'
    }

    actions_dict = {
        0: 'left',
        1: 'forward',
        2: 'right'
    }
    
    state = 0


    while(done==False):
        env.last_action = actions_dict[policy_iteration.policy[state]]
        state, rew , done = policy_iteration.step(state, policy_iteration.policy[state])
        cell = state // DIRECTIONS
        direct = state % DIRECTIONS
        car_x, car_y = get_coord(grid_size=grid_size, cell = cell)
        
        env.last_reward = rew
        env.last_done = done
        env.step(car_x, car_y, directions_dict[direct])
    # next_state, reward, done, _ = env.step('right')
    # env.render()

    # for _ in range(10):
    #     action = 'forward'
    #     next_state, reward, done, _ = env.step(action)
    #     env.render()
    time.sleep(1)
    # Quit Pygame
    pygame.quit()
