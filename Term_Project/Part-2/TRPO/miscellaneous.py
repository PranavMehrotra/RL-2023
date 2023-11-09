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