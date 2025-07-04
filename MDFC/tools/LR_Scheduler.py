###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/tools/script_utils.py
#       First Authors: David Kovacs, Ilyes Batatia
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import torch

class LRScheduler:
    """
    Tool to adjusts the learning rate during training to improve convergence and performance, typically by reducing it when progress stalls.
    Everything is handled by Torch
    """
    def __init__(self, optimizer, config) -> None:
        self.scheduler = config.scheduler
        self._optimizer_type = (config.optimizer)  # Schedulefree does not need an optimizer but checkpoint handler does.
        if config.scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=config.lr_scheduler_gamma
            )
        elif config.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=config.lr_factor,
                patience=config.scheduler_patience,
            )
        else:
            raise RuntimeError(f"Unknown scheduler: '{config.scheduler}'")

    def step(self, metrics=None, epoch=None):
        if self._optimizer_type == "schedulefree":
            return  # In principle, schedulefree optimizer can be used with a scheduler but the paper suggests it's not necessary
        if self.scheduler == "ExponentialLR":
            self.lr_scheduler.step(epoch=epoch)
        elif self.scheduler == "ReduceLROnPlateau":
            self.lr_scheduler.step( 
                metrics=metrics, epoch=epoch
            )

    def __getattr__(self, name):
        if name == "step":
            return self.step
        return getattr(self.lr_scheduler, name)
