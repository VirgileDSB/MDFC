###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/tools/scripts_utils.py
#       First Authors: David Kovacs, Ilyes Batatia
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import torch
from ...tools.config_tools import config_class_training


def get_optimizer(
    config: config_class_training, 
    model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Return an optimizer.
    Apply weight decay to the model parameters.
    """

    #####################################
    #
    # Weight decay is a penalty term added to the loss function if the model has large weights.
    # It is used to prevent overfitting and help the model generalize better.
    #
    #####################################

    decay_interactions = {}
    no_decay_interactions = {}

    for name, param in model.interactions.named_parameters(): #List of parameters in self.interactions in model (see model)
        # In MACE only these 2 weights have Weight decay applied so I kept it that way
        if "linear.weight" in name or "skip_tp_full.weight" in name: #linear is all cuet.linear and skip_tp_full is all cuet.ChannelWiseTensorProduct
            decay_interactions[name] = param
        else:
            no_decay_interactions[name] = param

    # This dict contains all parameters that may be used by a torch.optim class
    param_options = dict(
        params=[ #This is to aplly Weights decay 
            {
                "name": "embedding", # Useless parameter used for the user iformation only
                "params": model.node_embedding.parameters(), # List of the model parameters to optimize that will have this "name" and "weight_decay"
                "weight_decay": 0.0, # Is a Weight decay apply to thoses parameters (0 = no)
            },
            {
                "name": "interactions_decay", 
                "params": list(decay_interactions.values()), 
                "weight_decay": config.weight_decay, 
            },
            {
                "name": "interactions_no_decay",
                "params": list(no_decay_interactions.values()),
                "weight_decay": 0.0,
            },
            {
                "name": "products",
                "params": model.products.parameters(),
                "weight_decay": config.weight_decay,
            },
            {
                "name": "readouts",
                "params": model.readouts.parameters(),
                "weight_decay": 0.0,
            },
        ],
        lr=config.lr,
        amsgrad=config.amsgrad, # For adam and AdamW, if True, amsgrad methode is applied for better convergence
        betas=(config.beta, 0.999), # Betas is running average controle for optimisation "memory"
    )

    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(**param_options)
    elif config.optimizer == "schedulefree":
        try:
            from schedulefree import adamw_schedulefree
        except ImportError as exc:
            raise ImportError("`schedulefree` is not installed. Please install it via `pip install schedulefree` or `pip install mace-torch[schedulefree]`") from exc
        _param_options = {k: v for k, v in param_options.items() if k != "amsgrad"}
        optimizer = adamw_schedulefree.AdamWScheduleFree(**_param_options)
    else:
        optimizer = torch.optim.Adam(**param_options)
    return optimizer
