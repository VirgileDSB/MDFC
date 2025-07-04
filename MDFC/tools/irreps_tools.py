###########################################################################################
#
# Training script from github.com/ACEsuit/mace/mace/tools/train.py
#       First Authors: Ilyes Batatia, Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

from typing import List, Tuple
import cuequivariance as cue
import torch
from e3nn.util.jit import compile_mode

def tp_out_irreps(
    irreps1: cue.Irreps, 
    irreps2: cue.Irreps, 
    target_irreps: cue.Irreps
) -> Tuple[cue.Irreps, List]:
    """"
    tensor product of 2 irreps only keeping Irreps contained in target_irreps. \n
    """

    irreps_out_list: List[Tuple[int, cue.Irreps]] = []
    for mul, ir_in1 in irreps1: # If irreps is 16*0e + 4*1e : 1st loop is [16, 0e], and second will be [4, 1e]
        for _, ir_in2 in irreps2: # If irreps is 16*0e + 4*1e : ir_in2 -> 0e - 1e
            for ir_out in ir_in1 * ir_in2:  # Irrpes multiplication can give multiple irreps
                if ir_out in target_irreps:
                    irreps_out_list.append((mul, ir_out))

    # sort the irreps
    irreps_out = cue.Irreps("O3", irreps_out_list)
    irreps_out, _, _ = irreps_out.sort() # Output 2 and 3 are information about the sorting

    return irreps_out


def linear_out_irreps(
        irreps: cue.Irreps, 
        target_irreps: cue.Irreps
    ) -> cue.Irreps:
    """
    return target irreps contained in input irreps
    """
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f"{ir_in} not in {target_irreps}")

    return cue.Irreps("O3", irreps_mid)

@compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, 
        irreps: cue.Irreps, 
        cueq_config: str
    ) -> None:
        super().__init__()
        self.cueq_config = cueq_config
        self.irreps = irreps
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, 
        tensor: torch.Tensor
    ) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            if self.cueq_config == "mul_ir":
                field = field.reshape(batch, mul, d)
            else:
                field = field.reshape(batch, d, mul)
            out.append(field)

        if self.cueq_config == "mul_ir":
            return torch.cat(out, dim=-1)
        return torch.cat(out, dim=-2)
