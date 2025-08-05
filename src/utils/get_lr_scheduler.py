import torch.optim as optim
from .get_config_value import get_config_value


def exp_growth_lr_lambda(steps, exp_rate):
    return exp_rate ** steps


def get_lr_scheduler(optimizer, config):
    lr_method = get_config_value(config, "lr_method")

    if lr_method == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif lr_method == "exp_growth":
        exp_rate = get_config_value(config, "lr_exp_rate")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: exp_growth_lr_lambda(steps, exp_rate=exp_rate))
    else:
        raise ValueError(f"Unknown learning rate method: {lr_method}")

    return scheduler
