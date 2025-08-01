import torch
import torch.nn as nn

def get_full_grad_list(model, trainloader, optimizer, device):
    parameters = [p for p in model.parameters() if p.requires_grad]
    total_samples = len(trainloader.dataset)

    backup_grads = [p.grad.clone() if p.grad is not None else None for p in parameters]

    full_grad_list = [torch.zeros_like(p, device=device) for p in parameters]

    loss_fn = nn.CrossEntropyLoss(reduction='sum')

    for xx, yy in trainloader:
        xx, yy = xx.to(device), yy.to(device)
        optimizer.zero_grad()
        outputs = model(xx)
        loss = loss_fn(outputs, yy)
        loss.backward()

        for i, p in enumerate(parameters):
            if p.grad is not None:
                full_grad_list[i] += p.grad.detach()

    for g in full_grad_list:
        g /= total_samples

    for p, g_back in zip(parameters, backup_grads):
        p.grad = None if g_back is None else g_back.clone()

    total_norm = torch.sqrt(sum(g.norm()**2 for g in full_grad_list)).item()
    return total_norm

