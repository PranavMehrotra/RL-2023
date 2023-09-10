# Group/Serial Number: 11 
# Author: Pranav Mehrotra, 20CS10085
# Project Code: TCV1
# Project Title: Controlling a Toy Car around a Grid [Version 1]


import numpy as np
import matplotlib.pyplot as plt


DIRECTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
ACTIONS = ['L', 'F', 'R']

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

def plot_state_values_heatmap(grid_size, values, placeholder):
    state_values = np.zeros((4, grid_size, grid_size))
    for i in range(len(values)):
        cell = i//4
        direction = i%4
        x,y = get_coord(grid_size, cell)
        state_values[direction, y, x] = values[i]
    
    for i in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(state_values[i], interpolation='nearest', cmap='hot')
        ax.set_title(f'State Values for Direction-{DIRECTIONS[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels(np.arange(1, grid_size + 1))
        ax.set_yticklabels(np.arange(1, grid_size + 1))
        for row in range(grid_size):
            for col in range(grid_size):
                text = ax.text(col, row, round(state_values[i, row, col], 2), ha="center", va="center", color="blue", fontsize=15)
                
        # plt.show()
        plt.savefig(f'{placeholder}_state_values_heatmap_{DIRECTIONS[i]}.png')
        plt.close()
    
    print("\nState values heatmaps saved in the current directory\n")

def plot_optimal_policy_grid(grid_size, policy, placeholder):
    optimal_actions = np.zeros((4, grid_size, grid_size), dtype=np.int32)
    for i in range(len(policy)):
        cell = i//4
        direction = i%4
        x,y = get_coord(grid_size, cell)
        optimal_actions[direction, y, x] = policy[i]
    
    for i in range(4):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(optimal_actions[i], interpolation='nearest')
        ax.set_title(f'Optimal Actions for Direction-{DIRECTIONS[i]}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticklabels(np.arange(1, grid_size + 1))
        ax.set_yticklabels(np.arange(1, grid_size + 1))
        # Write the meaning of L,F,R in the legend
        ax.text(0, -1, 'L: Left   ', ha="center", va="center", color="black", fontsize=30)
        ax.text(1, -1, 'F: Forward   ', ha="center", va="center", color="black", fontsize=30)
        ax.text(2, -1, 'R: Right', ha="center", va="center", color="black", fontsize=30)
        for row in range(grid_size):
            for col in range(grid_size):
                text = ax.text(col, row, ACTIONS[optimal_actions[i, row, col]], ha="center", va="center", color="black", fontsize=30)
                
        # plt.show()
        plt.savefig(f'{placeholder}_optimal_policy_grid_{DIRECTIONS[i]}.png')
        plt.close()
    
    print("\nOptimal policy grids saved in the current directory\n")
