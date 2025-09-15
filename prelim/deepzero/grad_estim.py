import torch

def grad_estim_fo(model):
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_dict[name] = param.grad.detach().clone()
    return grad_dict

def apply_grad(model, grad_dict):
    for name, param in model.named_parameters():
        if name in grad_dict:
            param.grad = grad_dict[name].clone()

def gradient_estimation(model, method="fo", prune_mask=None):
    if method not in ["fo"]:
        raise NotImplementedError

    grad_dict = grad_estim_fo(model)
    if method == "fo":
        if prune_mask is not None:
            for pname, pmask in prune_mask.items():
                if pname in grad_dict:
                    grad_dict[pname] = grad_dict[pname] * pmask
        return grad_dict
    
    else:
        raise NotImplementedError