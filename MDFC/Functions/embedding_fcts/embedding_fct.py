###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/modules/radial.py
#       First Authors: Ilyes Batatia, Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

from e3nn.util.jit import compile_mode
import torch
import numpy as np



@compile_mode("script")
class BesselBasis(torch.nn.Module):
    """
    Bessel functions calculation
    """

    def __init__(self, 
        r_max: float, 
        num_basis: int = 8, 
        trainable: bool = False #if True, the bessel weights are learnable features
    ):
        super().__init__()

        # Bessel weights can be learnable features. They are initialized to be equally spaced in the range [0, pi/r_max]
        bessel_weights = (
            np.pi / r_max * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
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

    def forward(self, 
        x: torch.Tensor
    ) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )
    


@compile_mode("script")
class PolynomialCutoff(torch.nn.Module):
    """
    Polynomial cutoff function that goes from 1 to 0 as x goes from 0 to r_max.
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, 
        r_max: float, 
        p=6
    ):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer("r_max", torch.tensor(r_max, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.calculate_envelope(x, self.r_max, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, 
        r_max: torch.Tensor, 
        p: torch.Tensor
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



@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    """
    Radial Embedding Block transforming a scalar distance into num_bessel Bessel functions.
    The polynomial cutoff is a 1→0 function to smooth the cutoff discontinuity.
    Not yet in an Irreps representation.
    """
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
    ):
        super().__init__()
        #todo: add Transform and bessel and distance_transform?
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]

