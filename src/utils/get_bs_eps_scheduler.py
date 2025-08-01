from optim.bs_scheduler import LambdaBS
from .get_config_value import get_config_value


def exp_growth_bs_lambda(steps, exp_rate):
    return exp_rate ** steps


def linear_growth_bs_lambda(steps, initial_bs, coefficient):
    return 1 + (coefficient * steps / initial_bs)


def get_bs_eps_scheduler(config, max_bs=None):
    """Returns the batch size and epsilon lambda function based on the configuration."""
    bs_method = get_config_value(config, "bs_method")

    if bs_method == "constant":
        bs_scheduler = LambdaBS(initial_bs=get_config_value(config, "bs"), bs_lambda=lambda epoch: 1)
        bs_step_type = None

    elif bs_method == "exp_growth":
        initial_bs = get_config_value(config, "bs")
        bs_scheduler = LambdaBS(initial_bs=initial_bs, max_bs=max_bs, bs_lambda=lambda steps: exp_growth_bs_lambda(steps, exp_rate=get_config_value(config, "bs_exp_rate")))
        bs_step_type = get_config_value(config, "bs_step_type")

    elif bs_method == "linear_growth":
        initial_bs = get_config_value(config, "bs")
        bs_scheduler = LambdaBS(initial_bs=initial_bs, max_bs=max_bs, bs_lambda=lambda steps: linear_growth_bs_lambda(steps, initial_bs=initial_bs, coefficient=get_config_value(config, "coefficient")))
        bs_step_type = get_config_value(config, "bs_step_type")

    else:
        raise ValueError(f"Unknown batch size method: {bs_method}")

    return bs_scheduler, bs_step_type
