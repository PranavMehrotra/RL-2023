import os
import time
import torch
import torch.nn as nn

import numpy as np

from contcar_env import ContinuousCarRadarEnv

class DQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=128):
        super(DQN, self).__init__()
        self.input_size = state_dim
        self.output_size = num_actions
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)
    
    def save(self, checkpoint_dir, checkpoint_name):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.state_dict(), os.path.join(checkpoint_dir, checkpoint_name))
        print(f'\nModel saved in {os.path.join(checkpoint_dir, checkpoint_name)}\n')
    
    def load(self, checkpoint_dir, checkpoint_name):
        self.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_name)))
        print(f'\nModel loaded from {os.path.join(checkpoint_dir, checkpoint_name)}\n')


class ExperienceReplay():
    def __init__(self, state_dim, max_size=10000):
        self.max_size = max_size
        self.current_size = 0
        self.curr_idx = -1
        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int8)
        self.rewards = np.zeros(self.max_size, dtype=np.int32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.bool)

    def add(self, state, action, reward, next_state, done):
        idx = self.curr_idx = (self.curr_idx + 1) % self.max_size
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        if self.current_size < self.max_size:
            self.current_size += 1

    def sample(self, batch_size):
        if self.current_size < batch_size:
            raise Exception(f'Not enough samples in the experience replay buffer. Current size: {self.current_size}, Batch size: {batch_size}')
        idx = np.random.choice(np.arange(self.current_size), size=batch_size, replace=False)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]
    

class DQNAgent():
    def __init__(self, state_dim, num_actions, hidden_size=128, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001, tau=0.01, batch_size=64, max_memory_size=10000):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        
        self.model = DQN(self.state_dim, self.num_actions, self.hidden_size)
        self.target_model = DQN(self.state_dim, self.num_actions, self.hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.experience_replay = ExperienceReplay(self.state_dim, self.max_memory_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float()
                q_values = self.model(state)
                return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.experience_replay.add(state, action, reward, next_state, done)
    
    def replay(self, should_update_target_model=True):
        if self.experience_replay.current_size < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.experience_replay.sample(self.batch_size)
        
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).long()
        rewards = torch.from_numpy(rewards).int()
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).bool()

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = next_q_values.max(1)[0]
        expected_q_values = rewards + (~dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if should_update_target_model:
            self.update_target_model()

    def update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # self.target_model.load_state_dict(self.model.state_dict())

    

def train_dqn_agent(env: ContinuousCarRadarEnv, agent: DQNAgent, num_episodes=1000, max_steps=1000, start_learning = 200, learning_freq = 4, updation_freq = 100, checkpoint_dir='checkpoints', checkpoint_name='checkpoint_'):
    start_time = time.time()
    rewards = []
    epsilons = []
    total_steps = 0
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            total_steps += 1
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # env.render()
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if done:
                break
            if total_steps > start_learning and total_steps % learning_freq == 0:
                if total_steps % updation_freq == 0:
                    agent.replay(True)
                else:
                    agent.replay(False)
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        print(f'Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.4f}')
        # Save checkpoint if 50 episodes have passed
        if (episode + 1) % 50 == 0:
            agent.model.save(checkpoint_dir, checkpoint_name + str(episode + 1) + '.pth')
    print(f'\nTraining completed in {(time.time() - start_time) / 60:.2f} minutes')
    # agent.model.save(checkpoint_dir, checkpoint_name)
    return rewards, epsilons

def run_model(env: ContinuousCarRadarEnv, agent: DQNAgent, max_steps, checkpoint_dir='checkpoints', checkpoint_name='checkpoint.pth'):
    agent.model.load(checkpoint_dir, checkpoint_name)
    state = env.reset()
    done = False
    steps = 0
    while not done and steps < max_steps:
        steps += 1
        env.render()
        action = agent.act(state)
        state, reward, done = env.step(action)
        # time.sleep(0.1)
    if done:
        print(f'Car reached destination in {steps} steps')
    else:
        print(f'Car could not reach destination')
    env.close()

if __name__ == '__main__':
    # Set random seed
    np.random.seed(53467)
    # Initialize environment
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    state_dim = env.observation.shape[0]
    num_actions = env.num_actions
    
    # Initialize DQN agent
    agent = DQNAgent(state_dim, num_actions, hidden_size=64, gamma=0.99, epsilon=0.003, epsilon_min=0.0001, epsilon_decay=0.96, learning_rate=1e-4, tau=5e-4, batch_size=128, max_memory_size=40000)
    # agent.model.load('checkpoints', 'best_checkpoint_3_200.pth')
    # agent.model.load('checkpoints', 'excel_checkpoint_1_200_50.pth')

    run_model(env, agent, 4000, checkpoint_dir='checkpoints', checkpoint_name='checkpoint_1_200_50_50.pth')
    exit()

    # Train DQN agent
    rewards, epsilons = train_dqn_agent(env, agent, num_episodes=50, max_steps=5000, start_learning = 200, learning_freq = 3, updation_freq=30, checkpoint_dir='checkpoints', checkpoint_name='checkpoint_1_200_50_')
    
    # Plot rewards and epsilons
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(20, 10))
    # plt.plot(rewards)
    # plt.title('Rewards')
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    # plt.savefig('rewards.png')
    # plt.close()
    
    # plt.figure(figsize=(20, 10))
    # plt.plot(epsilons)
    # plt.title('Epsilons')
    # plt.xlabel('Episode')
    # plt.ylabel('Epsilon')
    # plt.savefig('epsilons.png')
    # plt.close()
    
    # Test DQN agent
    # agent.model.load('checkpoints', 'checkpoint.pth')
    # state = env.reset()
    # done = False
    # while not done:
    #     env.render()
    #     action = agent.act(state)
    #     state, reward, done = env.step(action)
    #     # time.sleep(0.1)
    # env.close()
    # run_model(env, agent, 6000, checkpoint_dir='checkpoints', checkpoint_name='checkpoint_200.pth')
