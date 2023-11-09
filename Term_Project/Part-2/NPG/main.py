import torch
from torch.autograd import Variable as V
import numpy as np

from contcar_env import ContinuousCarRadarEnv
from models import *
from miscellaneous import *
from hyperparams import Hyperparameters as hp
from npg import *

class ExperienceReplay():
    def __init__(self):
        self.mem = []

    def add(self, state, action, reward, next_state, done):
        self.mem.append((state, action, reward, next_state, done))

    def all(self):
        # Pack all the states, actions, rewards, next_states, and dones into their own lists
        states, actions, rewards, next_states, dones = zip(*self.mem)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

# Class for implementing normalised states
class RunningState():
    def __init__(self, state_dim, clip=0):
        self.mean = np.zeros(state_dim)
        self.std = np.ones(state_dim)
        self.n = 0
        self.clip = clip
        
    def update(self, x):
        self.n += 1
        if self.n == 1:
            self.mean = x.copy()
        else:
            last_mean = self.mean.copy()
            self.mean += (x - self.mean) / self.n
            self.std += (x - self.mean) * (x - last_mean)
        
    def normalise(self, x):
        x = (x - self.mean)
        if self.n > 1:
            x /= (np.sqrt(self.std / (self.n - 1)) + 1e-8)
        else:
            x /= (self.mean + 1e-8)
        if self.clip > 0:
            x = np.clip(x, -self.clip, self.clip)
        return x
    
    def add(self, x):
        self.update(x)
        return self.normalise(x)

# test the pre-trained model stores at "NPG.pth"
def test_model(): 
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    state_dim = env.observation.shape[0]
    action_dim = env.num_actions
    policy = Actor(state_dim, action_dim)
    policy.load_state_dict(torch.load('NPG.pth'))
    running_state = RunningState((state_dim,),)
    state = env.reset()
    state = running_state.add(state)
    reward_sum = 0
    for t in range(6000):
        action = policy(V(torch.Tensor(state)))
        action = action.data[0].numpy()
        next_state, reward, done = env.step(np.argmax(action))
        next_state = running_state.add(next_state)
        mask = 1
        if done:
            mask = 0
        reward_sum += reward
        state = next_state
        env.render()
        if done:
            break
    print('Episode {}\tLast reward: {:.2f}'.format(episode, reward_sum))
    env.close()
    exit(0)

    


if __name__ == '__main__':
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    state_dim = env.observation.shape[0]
    action_dim = env.num_actions
    policy = Actor(state_dim, action_dim)
    value = Critic(state_dim)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=hp.actor_lr)
    value_optim = torch.optim.Adam(value.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    running_state = RunningState((state_dim,),)
    episode = 0

    while episode < hp.max_episode:
        memory = ExperienceReplay()
        num_steps = 0
        reward_batch = 0
        num_episodes = 0

        while num_steps < hp.batch_size:
            state = env.reset()
            state = running_state.add(state)
            reward_sum = 0
            for t in range(6000):
                action = policy(V(torch.Tensor(state)))
                action = action.data[0].numpy()
                next_state, reward, done = env.step(np.argmax(action))
                next_state = running_state.add(next_state)
                mask = 1
                if done:
                    mask = 0
                memory.add(state, np.argmax(action), reward, next_state, mask)
                reward_sum += reward
                state = next_state
                num_steps += 1
                if done:
                    break

            num_episodes += 1
            reward_batch += reward_sum

        episode += num_episodes
        reward_batch /= num_episodes
        train_model(policy, value, memory.mem, policy_optim, value_optim)

        print('Episode {}\tLast reward: {:.2f}\tAverage reward {:.2f}'.format(
            episode, reward_sum, reward_batch))
    # Save the model
    torch.save(policy.state_dict(), 'NPG.pth')
        
    env.close()



