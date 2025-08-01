import torch.optim as optim
from .get_config_value import get_config_value


def exp_growth_lr_lambda(steps, exp_rate):
    return exp_rate ** steps


def get_lr_scheduler(optimizer, config):
    lr_method = get_config_value(config, "lr_method")

    if lr_method == "constant":
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        lr_step_type = None
    elif lr_method == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_config_value(config, "epochs"), eta_min=config.get("lr_min", 0))
        lr_step_type = "epoch"
    elif lr_method == "exp_growth":
        exp_rate = get_config_value(config, "lr_exp_rate")
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: exp_growth_lr_lambda(steps, exp_rate=exp_rate))
        lr_step_type = get_config_value(config, "lr_step_type")
    else:
        raise ValueError(f"Unknown learning rate method: {lr_method}")

    return scheduler, lr_step_type
