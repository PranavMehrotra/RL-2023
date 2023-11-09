import numpy as np
from miscellaneous import *
from hyperparams import Hyperparameters as hp


def objective_function(actor, returns, states, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    log_policy = normal_log_density(torch.Tensor(actions), mu, std, logstd)
    returns = returns.unsqueeze(1)

    objective = returns * log_policy
    objective = objective.mean()
    return objective


def train_critic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            target = returns.unsqueeze(1)[batch_index]

            values = critic(inputs)
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()

def fisher_vector_product(actor, states, p):
    p.detach()
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = get_gradients(kl_grad)  # check kl_grad == 0

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + 0.1 * p

def get_returns(rewards, mask):
    rewards = torch.Tensor(rewards)
    mask = torch.Tensor(mask)
    returns = torch.zeros_like(rewards)
    running_returns = 0
    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * mask[t]
        returns[t] = running_returns
    returns = (returns - returns.mean()) / returns.std()
    return returns

def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 4])

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks)

    # ----------------------------
    # step 2: train critic several steps with respect to returns
    train_critic(critic, states, returns, critic_optim)

    # ----------------------------
    # step 3: get gradient of loss and hessian of kl
    loss = objective_function(actor, returns, states, actions)
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = get_gradients(loss_grad)
    step_dir = conjugate_gradients(actor, states, loss_grad.data, nsteps=10)

    # ----------------------------
    # step 4: get step direction and step size and update actor
    params = get_params(actor)
    new_params = params + 0.5 * step_dir
    set_params(actor, new_params)