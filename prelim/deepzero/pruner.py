import torch

def importance_score_grasp(model, train_loader, criterion):
    model.eval()

    num_samples = 128
    grasp_step = 5e-3

    num_processed = 0
    gradient_sum = {pname: torch.zeros_like(param) for pname, param in model.named_parameters() if param.requires_grad}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for pname, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_sum[pname] += param.grad.detach().clone() * data.size(0)
                param.grad.zero_()
        num_processed += data.size(0)

        if (batch_idx+1)*data.size(0) >= num_samples:
            break
    
    g0 = {pname: (grad / num_processed) for pname, grad in gradient_sum.items()}
    for pname, param in model.named_parameters():
        if pname in g0:
            param.data.add_(grasp_step * torch.sign(g0[pname]))

    num_processed = 0
    gradient_sum = {pname: torch.zeros_like(param) for pname, param in model.named_parameters() if param.requires_grad}

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(next(model.parameters()).device), target.to(next(model.parameters()).device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        for pname, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradient_sum[pname] += param.grad.detach().clone() * data.size(0)
                param.grad.zero_()
        num_processed += data.size(0)

        if (batch_idx+1)*data.size(0) >= num_samples:
            break
    
    g1 = {pname: (grad / num_processed) for pname, grad in gradient_sum.items()}
    for pname, param in model.named_parameters():
        if pname in g0:
            param.data.sub_(grasp_step * torch.sign(g0[pname]))

    Hg = {pname: (g1[pname] - g0[pname]) / grasp_step for pname in g0.keys()}
    score = {pname: (param.data * Hg[pname]) for pname, param in model.named_parameters() if pname in Hg}

    return score

def gmask_lpr(model, score, prune_ratio=0.9):
    all_scores = torch.cat([torch.flatten(torch.abs(v)) for v in score.values()])
    threshold, _ = torch.kthvalue(all_scores, int(len(all_scores) * prune_ratio))

    grad_mask = {}
    for pname, param in model.named_parameters():
        if pname in score and 'bias' not in pname:
            grad_mask[pname] = (torch.abs(score[pname]) >= threshold).float()
            grad_mask[pname] = grad_mask[pname].flatten()[torch.randperm(grad_mask[pname].numel())].view(grad_mask[pname].shape)
        elif pname in score and 'bias' in pname:
            grad_mask[pname] = torch.ones_like(param)

    return grad_mask