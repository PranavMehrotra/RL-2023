import scipy.optimize
import torch
from torch.autograd import Variable as V
import numpy as np

from contcar_env import ContinuousCarRadarEnv
from models import *
from miscellaneous import *
from trpo import step

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


class hyper_params():
    def __init__(self, gamma=0.995, tau=0.97, l2_reg=0.001, max_kl=0.01, damping=0.1, seed=65431, batch_size=14000, num_episodes=10, hidden_size=64, render=False):
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping
        self.seed = seed
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.hidden_size = hidden_size
        self.render = render

def select_action(policy, state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy(V(state))
    return torch.normal(action_mean, action_std)

def test_model(check_point_path, env: ContinuousCarRadarEnv, num_inputs, max_steps=4000):
    policy = Policy(env.observation.shape[0], env.num_actions)
    policy.load_state_dict(torch.load(check_point_path))
    policy.eval()
    state = env.reset()
    running_state = RunningState((num_inputs,))
    state = running_state.add(state)
    done = False
    reward = 0
    steps = 0
    while not done and steps < max_steps:
        action = select_action(policy, state).data[0].numpy()
        next_state, r, done = env.step(np.argmax(action))
        next_state = running_state.add(next_state)
        reward += r
        steps += 1
        state = next_state
        env.render()

    print(f'Total reward: {reward}')
    env.close()
    exit(0)

if __name__ == '__main__':
    hp = hyper_params(num_episodes=18)
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    num_inputs = env.observation.shape[0]
    num_actions = env.num_actions

    env.set_seed(hp.seed)
    torch.manual_seed(hp.seed)

    # running_state = RunningState((num_inputs,))

    # Test pretrained model
    test_model('trpo.pth', env, num_inputs)

    policy = Policy(num_inputs, num_actions, hp.hidden_size)
    value = Value(num_inputs, hp.hidden_size)


    # def select_action(state):
    #     state = torch.from_numpy(state).unsqueeze(0)
    #     action_mean, _, action_std = policy(V(state))
    #     return torch.normal(action_mean, action_std)

    def update_params(batch):
        states = batch[0]
        actions = batch[1]
        rewards = batch[2]
        masks = batch[4]
        values = value(V(states))

        advantages = torch.Tensor(actions.size(0),1)
        deltas = torch.Tensor(actions.size(0),1)
        returns = torch.Tensor(actions.size(0),1)

        prev_advantage = 0
        prev_return = 0
        prev_value = 0
        for i in reversed(range(rewards.size(0))):
            returns[i] = rewards[i] + hp.gamma * prev_return * masks[i]
            deltas[i] = rewards[i] + hp.gamma * prev_value * masks[i] - values.data[i]
            advantages[i] = deltas[i] + hp.gamma * hp.tau * prev_advantage * masks[i]

            prev_advantage = advantages[i, 0]
            prev_return = returns[i, 0]
            prev_value = values.data[i, 0]

        targets = V(returns)

        def get_value_loss(params):
            set_params(value, torch.Tensor(params))
            for param in value.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)
            
            _values = value(V(states))
            value_loss = (_values - targets).pow(2).mean()

            for param in value.parameters():
                value_loss += param.pow(2).sum() * hp.l2_reg
            value_loss.backward()
            return (value_loss.data.double().numpy(), get_gradients(value).data.double().numpy())
        
        params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_params(value).double().numpy(), maxiter=25)
        set_params(value, torch.Tensor(params))
        advantages = (advantages - advantages.mean()) / advantages.std()
        action_means, action_log_stds, action_stds = policy(V(states))
        fixed_log_probs = normal_log_density(V(actions), action_means, action_log_stds, action_stds).data.clone()

        def objective_function(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy(V(states))
            else:
                action_means, action_log_stds, action_stds = policy(V(states))
            log_probs = normal_log_density(V(actions), action_means, action_log_stds, action_stds)
            action_loss = torch.exp(log_probs - V(fixed_log_probs)) * -V(advantages)

            return action_loss.mean()
        
        def kl_divergence_function():
            mean, log_std, std = policy(V(states))

            temp_mean, temp_log_std, temp_std = V(mean.data), V(log_std.data), V(std.data)
            kl = log_std - temp_log_std + (temp_std.pow(2) + (temp_mean - mean).pow(2)) / (2.0 * std.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)
        
        step(policy, objective_function, kl_divergence_function, hp.max_kl, hp.damping)

    for i in range(hp.num_episodes):
        experience = ExperienceReplay()
        steps = 0
        reward = 0
        runs = 0
        while steps < hp.batch_size:
            state = env.reset()
            running_state = RunningState((num_inputs,))
            state = running_state.add(state)
            cumul_reward = 0
            
            for j in range(7000):
                action = select_action(policy, state).data[0].numpy()
                next_state, r, done = env.step(np.argmax(action))
                next_state = running_state.add(next_state)
                cumul_reward += r
                mask = 1
                if done:
                    mask = 0
                experience.add(state, action, r, next_state, mask)
                if hp.render:
                    env.render()
                if done:
                    break
                state = next_state  
            steps += (j+1)
            reward += cumul_reward
            runs += 1

        batch = experience.all()
        reward /= runs
        update_params(batch)

        print('Episode: {}\tReward: {:.2f}'.format(i, reward))

    torch.save(policy.state_dict(), 'trpo.pth')

    env.close()

