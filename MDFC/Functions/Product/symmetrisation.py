###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/modules/blocks.py
#       First Authors: Ilyes Batatia, Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

from typing import Optional
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode
from ...tools.config_tools import config_class_training
import cuequivariance_torch as cuet
import cuequivariance as cue


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    """
    symmetric contractions of the output of the interaction block
    """
    def __init__(
        self,
        node_feats_irreps: cue.Irreps,
        target_irreps: cue.Irreps,
        config: config_class_training,
        use_sc: bool,
        correlation: int
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.layout_str = config.layout

        # symmetrising message as explained in the paper (https://arxiv.org/abs/2206.07697 E.10) 
        self.symmetric_contractions = cuet.SymmetricContraction(
            cue.Irreps("O3", node_feats_irreps),
            cue.Irreps("O3", target_irreps),
            layout_in=cue.ir_mul,
            layout_out=config.layout,
            contraction_degree=correlation,
            num_elements=config.num_elements,
            use_fallback = True, #Originaly False but fix a kernel error ; todo: Get rid of that??
            original_mace=True, #mysterious option that change the way the output is calculated, I keept it as originaly in mace
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
        )

        # linear layer
        self.linear = cuet.Linear(
            target_irreps,
            target_irreps,
            use_fallback=True, #todo: same, False?
            internal_weights=True,
            shared_weights=True,
            layout=config.layout,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
                
        if self.layout_str == "mul_ir":
            node_feats = torch.transpose(node_feats, 1, 2)
        index_attrs = torch.nonzero(node_attrs)[:, 1].int()
        node_feats = self.symmetric_contractions(
            node_feats.flatten(1),
            index_attrs,
        )

        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)

