import os
import torch
import numpy as np

from contcar_env import ContinuousCarRadarEnv
from ppo import PPO

def test_model():
    ############## Hyperparameters ##############
    # creating environment
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    state_dim = env.observation.shape[0]
    action_dim = env.num_actions
    render = True
    num_test_episodes = 1
    
    # ppo hyperparameters
    K_epochs = 50
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    random_seed = 37373
    #############################################

    ## training loop
    env.set_seed(random_seed)
    torch.manual_seed(random_seed)

    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    ppo.policy.load_state_dict(torch.load(os.path.join(os.getcwd(), 'PPO_trained.pth')))
    running_reward = 0
    timestep = 0
    episode = 0
    while episode < num_test_episodes:
        episode += 1
        state = env.reset()
        current_episode_reward = 0
        for t in range(10000):
            timestep += 1
            action = ppo.select_action(state)
            state, reward, done = env.step(action)
            current_episode_reward += reward

            if render:
                env.render()
            if done:
                break

        running_reward = 0.05 * current_episode_reward + (1 - 0.05) * running_reward
        print(f'Episode {timestep}\tLast reward: {current_episode_reward:.2f}\tAverage reward: {running_reward:.2f}')

    env.close()
    exit()

if __name__ == '__main__':
    test_model()  ##### Comment this line to train the model #####
    ############## Hyperparameters ##############
    # creating environment
    env = ContinuousCarRadarEnv(reward_inside_path=1, reward_outside_path=-3, car_speed=1.3, outer_ring_radius=350, inner_ring_radius= 150)
    state_dim = env.observation.shape[0]
    action_dim = env.num_actions
    render = False
    
    # training hyperparameters
    max_episode_len = 6000
    max_training_timestep = 210000

    # ppo hyperparameters
    update_timestep = max_episode_len
    K_epochs = 50
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.0003
    lr_critic = 0.001
    random_seed = 33985
    #############################################

    ## training loop
    env.set_seed(random_seed)
    torch.manual_seed(random_seed)

    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    running_reward = 0
    timestep = 0

    while timestep <= max_training_timestep:
        state = env.reset()
        current_episode_reward = 0
        for t in range(max_episode_len):
            timestep += 1
            action = ppo.select_action(state)
            state, reward, done = env.step(action)
            current_episode_reward += reward
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                ppo.update()

            if render:
                env.render()
            if done:
                break

        running_reward = 0.05 * current_episode_reward + (1 - 0.05) * running_reward
        print(f'Episode {timestep}\tLast reward: {current_episode_reward:.2f}\tAverage reward: {running_reward:.2f}')

    torch.save(ppo.policy.state_dict(), os.path.join(os.getcwd(), 'PPO.pth'))
    env.close()
