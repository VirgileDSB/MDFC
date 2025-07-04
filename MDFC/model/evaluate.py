###########################################################################################
#
# Training script from github.com/ACEsuit/mace/mace/tools/train.py
#       First Authors: Ilyes Batatia, Gregor Simm, David Kovacs
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import time
from typing import Any, Dict, Tuple
from tqdm import tqdm
from torchmetrics import Metric

from ..tools import torch_geometric
from ..tools.torch_tools import to_numpy

from typing import List, Union
import numpy as np


import torch
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    metrics_to_print: List[str],
) -> Tuple[float, Dict[str, Any]]:
    """
    evaluate model \n
    output is: \n
        avg_loss, 
        aux metrics
    """

    # For evaluation, desactivate requires_grad
    for param in model.parameters():
        param.requires_grad = False

    # Metrics class defined below
    compute_metrics = Metrics_class(loss_fn=loss_fn, metrics_to_print = metrics_to_print).to(device)

    # Keep track of the time for model evaluation
    start_time = time.time()

    print("")
    print(f"\033[34m Evaluating \033[0m")

    # Run the model
    i=0
    for batch in tqdm(data_loader):
        i
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(batch_dict, training=False)

        # strore infos of batch in metrics class (This line is processing "update" method of the Metrics class)
        compute_metrics(batch, output) # Please feel free to add any metrics you want to compute here

    # Calculate and output the metrics
    avg_loss, metrics = compute_metrics.compute()
    metrics["time"] = time.time() - start_time

    # "Metric" class precoded method
    compute_metrics.reset()

    # reactivate requires_grad
    for param in model.parameters():
        param.requires_grad = True

    # return metrics
    return avg_loss, metrics


class Metrics_class(Metric):
    """ 
    Metrics child class: supports multiprocessing, provides access to 'add_state', 'update()', and 'compute()'. \n
    Class to keep track of the training metrics during evaluation, providing access to: \n
    --> update: 'forward method of the Metric class', updates the metric information \n
    --> compute: returns computed loss, energy, and force statistics.
    """
    def __init__(
            self, 
            loss_fn: torch.nn.Module,
            metrics_to_print: List[str]
        ):
        super().__init__()
        self.loss_fn = loss_fn
        self.metrics_to_print = metrics_to_print

        #####################################################################
        #
        # 'add_state' creates an object in self that tracks changes in the variable it points to.
        # Any future changes to the variable in the class will be saved.
        # Compatible with multiprocessing using 'dist_reduce_fx'.
        #
        #####################################################################

        self.add_state("total_running_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_data", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("forces_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("forces", default=[], dist_reduce_fx="cat")
        self.add_state("forces_difference", default=[], dist_reduce_fx="cat")

        # I disabled the energy part because I doubt that it is useful. Can be reimplemented for testing in the future.
        #self.add_state("E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum")
        #self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        #self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")


    def update(
            self, 
            batch: torch_geometric.batch.Batch, 
            output: Dict[str, torch.Tensor],
        ): 
        """ 
        'update' is 'forward methode' of 'Metric class' \n
        -> calculate loss \n
        -> update metrics
        """

        loss = self.loss_fn( # The only loss function currently implemented is WeightedEnergyForcesLoss; others can be added.
            output=output, 
            Batch=batch, 
        )

        # Saving Loss
        self.total_running_loss += loss
        self.num_data += torch.tensor(1.0)
        
        # Force mectrics
        self.forces_computed += 1.0
        self.forces.append(batch["snapshot_particules_force"])
        self.forces_difference.append(batch["snapshot_particules_force"] - output["forces"])

        # Energy metrics
        # self.E_computed += 1.0
        # self.delta_es.append(batch["snapshot_total_energy"] - output["snapshots_energy"])
        # self.delta_es_per_atom.append((batch["snapshot_total_energy"] - output["snapshots_energy"]) / (batch.ptr[1:] - batch.ptr[:-1]))

        # Add more metrics:

    def convert(self, delta: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return to_numpy(delta)

    def compute(self): #todo: check that
        """
        Return computed loss energy and force stats
        """
        aux = {}
        aux["loss"] = to_numpy(self.total_running_loss / self.num_data).item()

        # Compute metrics
        forces = self.convert(self.forces)
        forces_difference = self.convert(self.forces_difference)

        if "percentile" in self.metrics_to_print:
            aux["percentile"] = self.compute_percentile_95(forces_difference)

        if "relativeRMSE" in self.metrics_to_print:
            aux["relativeRMSE"] = self.compute_rel_rmse(forces_difference, forces)

        if "RMSE" in self.metrics_to_print:
            aux["RMSE"] = self.compute_rmse(forces_difference)

        if "MAE" in self.metrics_to_print:
            aux["MAE"] = self.compute_mae(forces_difference)

        if "relativeMAE" in self.metrics_to_print:
            aux["relativeMAE"] = self.compute_rel_mae(forces_difference, forces)

        # Disabeled energy metrics calculations
        #if self.E_computed:
        #    delta_es = self.convert(self.delta_es)
        #    delta_es_per_atom = self.convert(self.delta_es_per_atom)
        #    aux["mae_e"] = compute_mae(delta_es)
        #    aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
        #    aux["rmse_e"] = compute_rmse(delta_es)
        #    aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
        #    aux["q95_e"] = compute_q95(delta_es)

        return aux["loss"], aux
    
    def compute_percentile_95(self, delta: np.ndarray) -> float:
        """
        Compute the 95th percentile of the absolute values of the delta array.
        """
        return np.percentile(np.abs(delta), q=95)
    
    def compute_mae(self, delta: np.ndarray) -> float:
        """ compute mean absolute error """
        return np.mean(np.abs(delta)).item()
    
    def compute_rel_mae(self, delta: np.ndarray, target_val: np.ndarray) -> float:
        """ compute relative mean absolute error """
        target_norm = np.mean(np.abs(target_val))
        return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100
    
    def compute_rmse(self, delta: np.ndarray) -> float:
        """ compute root mean square error """
        return np.sqrt(np.mean(np.square(delta))).item()
    
    def compute_rel_rmse(self, delta: np.ndarray, target_val: np.ndarray) -> float:
        """ compute relative root mean square error """
        target_norm = np.sqrt(np.mean(np.square(target_val))).item()
        return np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100

