import numpy as np
import torch
import math

def get_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)

def set_params(model, params):
    i = 0
    for param in model.parameters():
        param.data.copy_(params[i:i+param.numel()].view_as(param))
        i += param.numel()

def get_gradients(model):
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad.view(-1))
    return torch.cat(gradients)

def normal_log_density(x, mean, log_std, std):
    variance = std.pow(2)
    log_density = -(x - mean).pow(2) / (2 * variance) - 0.5 * math.log(2 * math.pi) - log_std
    return log_density.sum(1, keepdim=True)

def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def conjugate_gradients(Fvp, b, n, residual=1e-10):
    y = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(n):
        fvp_val = Fvp(p)
        alpha = rdotr / torch.dot(p, fvp_val)
        y += alpha * p
        r -= alpha * fvp_val
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual:
            break
    return y