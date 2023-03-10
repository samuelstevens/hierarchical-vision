import composer.optim
import torch.optim


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}

    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    parameters = set_weight_decay(model, skip)

    name = config.optim.name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            momentum=config.optim.momentum,
            nesterov=config.optim.nesterov,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    elif name == "decoupledadamw":
        return composer.optim.DecoupledAdamW(
            parameters,
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    elif name == "decoupledsgdw":
        return composer.optim.DecoupledSGDW(
            parameters,
            lr=config.optim.lr,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
        )
    else:
        raise ValueError(name)


def set_weight_decay(model, skip_list=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list):
            no_decay.append(param)
        else:
            has_decay.append(param)

    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]
