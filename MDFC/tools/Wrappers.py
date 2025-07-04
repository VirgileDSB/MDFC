###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
#
###########################################################################################


import cuequivariance_torch as cuet
import cuequivariance as cue
from e3nn import nn
from e3nn.util.jit import compile_mode
from .compile import simplify_if_compile
import torch
from .config_tools import config_class_training
import numpy as np
from typing import Dict, Optional, Callable

# The ScaleShiftBlock is actually commented out ine the model code because it was causing a error one matrices size mismatch.
# The error may not be complicated to fix, but I did not have time to do it before the deadline.
# Please consider adding it back as soon as possible, it may really improve the model performance.
@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(
        self, 
        shift: float
    ):
        super().__init__()

        self.register_buffer("shift", torch.tensor(shift, dtype=torch.get_default_dtype()))

    def forward(
            self, 
            x: torch.Tensor, 
        ) -> torch.Tensor:
        
        return (x + torch.ones(len(x), device=x.device) * self.shift)

    def __repr__(self):
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(shift={formatted_shift})"


@simplify_if_compile
@compile_mode("script")
class Non_Linear_wrapper(torch.nn.Module):
    """
    torch module class. Wrapper around : cuet.Linear + activation + cuet.Linear (Irreps perceptron) \n
    """
    def __init__(
        self,
        irreps_in: cue.Irreps,
        MLP_irreps: cue.Irreps,
        config: config_class_training,
        irrep_out: cue.Irreps,
    ):
        super().__init__()

        gate_dict: Dict[str, Optional[Callable]] = {
            "abs": torch.abs,
            "tanh": torch.tanh,
            "silu": torch.nn.functional.silu,
            "None": None,
        }

        self.linear_1 = cuet.Linear(
            irreps_in, 
            MLP_irreps, 
            layout=config.layout,
            use_fallback = True
        )

        self.non_linearity = nn.Activation(
            irreps_in=str(MLP_irreps), 
            acts=[gate_dict[config.gate]]
        )

        self.linear_2 = cuet.Linear(
            MLP_irreps, 
            irrep_out, 
            layout=config.layout,
            use_fallback = True
        )

    def forward(
        self, 
        x: torch.Tensor, 
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.linear_1(x)
        x = self.non_linearity(x)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class Linear_wrapper(torch.nn.Module):
    """
    torch module class. Wrapper around cuet.Linear \n
    cuet.Linear have internal auto-managed weight tensor \n
    cuet.Linear auto-manage irreps
    """
    def __init__(
        self,
        irreps_in: cue.Irreps,
        irreps_out: cue.Irreps,
        config: config_class_training,
        shared_weights: bool = True,
    ):
        super().__init__()

        self.linear = cuet.Linear(
            irreps_in, 
            irreps_out, 
            layout=config.layout,
            shared_weights=shared_weights,
            use_fallback=True
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class radial_to_Bessel(torch.nn.Module):
    """
    torch module class. Wrapper around a manual bessel transformation \n
    used for embedding \n 
    Bessel numbers are learnable features, initialised as [1, 2, 3, ... , num_bessel] \n
    One can set bessel numbers as non leanable registered features with trainable = False \n    
    __repr__ to see bessel_weights values
    """

    def __init__(self, r_max: float, num_bessel=8, trainable=True):
        super().__init__()

        #
        bessel_weights = (np.pi/ r_max * torch.linspace(start=1.0, end=num_bessel, steps=num_bessel, dtype=torch.get_default_dtype(),))


        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_bessel]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_bessel={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


#todo: add more cutoff types (learnable shift?)
@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """
    torch module class. Wrapper around polynomial transformation \n
    ---> soft cutoff sigmoid-like function that goes from 1 to 0 as x goes from 0 to r_max. \n
    used juste before bessel transformation to soften cuttoff stepp fct \n
    ---> no leanable features in this transformation 
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"