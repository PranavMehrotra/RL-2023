import numpy as np
import torch
from torch.autograd import Variable as V
from miscellaneous import *

def line_search(model, objective_function, current_parameters, full_step_direction, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    # Evaluate the objective function at the current parameters
    function_value = objective_function(True).data

    # Loop over potential step fractions
    for(_, step_fraction) in enumerate(.5**np.arange(max_backtracks)):
        # Calculate the new parameters
        new_parameters = current_parameters + step_fraction * full_step_direction

        # Set the model parameters to the new parameters
        set_params(model, new_parameters)

        # Evaluate the objective function at the new parameters
        new_function_value = objective_function(True).data

        actual_improvement = function_value - new_function_value
        expected_improvement = expected_improve_rate * step_fraction
        ratio = actual_improvement / expected_improvement

        # If the improvement is sufficient
        if ratio.item() > accept_ratio and actual_improvement.item() > 0:
            return True, new_parameters

    # If no suitable step was found
    return False, current_parameters

def step(model, objective_function, kl_divergence_function, max_kl, damping):
    # Evaluate the objective function
    objective_value = objective_function()

    # Calculate the gradients of the objective function with respect to the model parameters
    grads = torch.autograd.grad(objective_value, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    # Function to calculate the product of the Fisher information matrix and a vector
    def fisher_vector_product(vector):
        kl_divergence = kl_divergence_function()
        kl_divergence = kl_divergence.mean()

        # Calculate the gradients of the KL divergence with respect to the model parameters
        grads = torch.autograd.grad(kl_divergence, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        # Calculate the product of the Fisher information matrix and the vector
        kl_vector_product = (flat_grad_kl * V(vector)).sum()
        grads = torch.autograd.grad(kl_vector_product, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + vector * damping
    
    # Calculate the product of the Fisher information matrix and the gradient of the objective function
    step_direction = conjugate_gradients(fisher_vector_product, -loss_grad, 10)

    # Calculate the product of the step direction and the Fisher information matrix
    shs = 0.5 * (step_direction * fisher_vector_product(step_direction)).sum(0, keepdim=True)

    # Calculate the Lagrange multiplier
    lagrange_multiplier = torch.sqrt(shs / max_kl)
    full_step = step_direction / lagrange_multiplier[0]

    # Calculate the dot product of the gradient of the objective function and the step direction
    neg_dot_product = (-loss_grad * step_direction).sum(0, keepdim=True)
    print(("Lagrange multiplier:", lagrange_multiplier[0], "Gradient norm:", loss_grad.norm()))

    # Update the model parameters
    current_parameters = get_params(model)
    success, new_parameters = line_search(model, objective_function, current_parameters, full_step, neg_dot_product / lagrange_multiplier[0])
    set_params(model, new_parameters)

    return objective_value

