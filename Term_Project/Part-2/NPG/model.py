import os
import time 
import torch
import torch.nn as nn
import numpy as np
from contcar_env import ContinuousCarRadarEnv

# Class for implementing Natural Policy Gradient
class NPG(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_size=64):
        super(NPG, self).__init__()
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
        idxs = np.random.randint(0, self.current_size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idxs]),
            torch.LongTensor(self.actions[idxs]),
            torch.FloatTensor(self.rewards[idxs]),
            torch.FloatTensor(self.next_states[idxs]),
            torch.FloatTensor(self.dones[idxs])
        )


class NPGAgent():
    def __init__(self, state_dim, num_actions, hidden_size=64, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate = 5e-4, batch_size=64, max_memory_size=10000):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size

        self.model = NPG(self.state_dim, self.num_actions, self.hidden_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model = NPG(self.state_dim, self.num_actions, self.hidden_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.tm_optimizer = torch.optim.Adam(self.target_model.parameters(), lr=self.learning_rate)

        self.memory = ExperienceReplay(self.state_dim, self.max_memory_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.target_model.to(self.device)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).float().to(self.device)
                logits = self.model(state)
                


    # def train_step(self):
    #     states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
    #     states = states.to(self.device)
    #     actions = actions.to(self.device)
    #     rewards = rewards.to(self.device)
    #     next_states = next_states.to(self.device)
    #     dones = dones.to(self.device)

    #     logits = self.model(states)
    #     log_probs = nn.functional.log_softmax(logits, dim=1)
    #     action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    #     next_logits = self.model(next_states)
    #     next_log_probs = nn.functional.log_softmax(next_logits, dim=1)
    #     next_action_log_probs = next_log_probs.gather(1, next_logits.argmax(dim=1).unsqueeze(1)).squeeze(1)
    #     expected_action_log_probs = rewards + (~dones) * self.gamma * next_action_log_probs
    #     loss = -torch.mean(action_log_probs * expected_action_log_probs)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()

    #     return loss.item()
    

    

if __name__ == '__main__':
    env = ContinuousCarRadarEnv()
    agent = NPGAgent(env, train=True)
    agent.train_model(num_episodes=1000, render=False)
    agent.model.save('models', 'npg.pth')
    env.close()
