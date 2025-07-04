###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/modules/loss.py
#       First Authors: Ilyes Batatia, Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

import torch
from typing import Dict

from ...tools.torch_geometric import Batch

# same same:

class euclidian_squared_error(torch.nn.Module):
    def __init__(self, train_loader) -> None:
        super().__init__()
        first_batch = next(iter(train_loader))
        self.mean_force_squared = torch.norm(first_batch["snapshot_particules_force"], dim=1).mean().pow(2)

    def forward(self, 
        Batch: Batch, 
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return euclidian_squared_error_force(Batch, output, self.mean_force_squared)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}")



class projected_mean_squared_error(torch.nn.Module):
    def __init__(self,) -> None:
        super().__init__()


    def forward(self, 
        Batch: Batch, 
        output: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return mean_squared_error_forces(Batch, output)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}")


def mean_squared_error_forces(
        Batch: Batch, 
        output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
    return torch.mean(torch.square(Batch["snapshot_particules_force"] - output["forces"]))

def euclidian_squared_error_force(
    Batch: Batch, 
    output: Dict[str, torch.Tensor],
    mean_Froce: float
) -> torch.Tensor:
    euclidian_vectors = Batch["snapshot_particules_force"] - output["forces"]
    euclidean_norms = torch.norm(euclidian_vectors, dim=1).pow(2)/(mean_Froce + 1e-8)
    return torch.mean(euclidean_norms)