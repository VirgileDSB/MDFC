###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/tools/train.py
#       First Authors: Ilyes Batatia, Gregor Simm, David Kovacs
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import time
from typing import Any, Dict, Optional, Tuple
import logging
logger = logging.getLogger("Main_Logger")

import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from ..tools import torch_geometric
from ..tools.torch_tools import to_numpy



def take_step(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    
    """ elementary step function: \n
    --> aply model(batch), calculate loss, optimizer.step \n
    "time" is is calculated for later speed check
    """
    
    start_time = time.time()
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    #def closure():
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch_dict,
        training=True,
    )
    loss = loss_fn( # The only loss function currently implemented is WeightedEnergyForcesLoss ; others can be added.
        output=output, 
        Batch=batch, 
    )
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

    optimizer.step()

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict


def take_step_lbfgs(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
) -> Tuple[float, Dict[str, Any]]:
    """ 'take_step' boosted with 'Limited memory Broyden Fletcher Goldfarb Shanno' quasi newton optimisation step technique for fast convergence\n
    'take_step': \n
    '\n
    elementary step function: \n
    --> aply model(batch), calculate loss, optimizer.step \n
    "time" is is calculated for later speed check\n
    '
    """
    start_time = time.time()
    logger.debug(
        f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB"
    )

    total_sample_count = 0
    for batch in data_loader:
        total_sample_count += batch.num_graphs

    signal = None

    def closure():
        """A closure function is needed for LBFGS optimiser (it's the way it need to be implemented)
        """

        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        # Process each batch and then collect the results we pass to the optimizer
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()
            output = model(
                batch_dict,
                training=True,
            )
            batch_loss = loss_fn(pred=output, ref=batch)
            batch_loss = batch_loss * (batch.num_graphs / total_sample_count)

            batch_loss.backward()
            total_loss += batch_loss

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        return total_loss


    loss = optimizer.step(closure)

    if ema is not None:
        ema.update()

    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    return loss, loss_dict

