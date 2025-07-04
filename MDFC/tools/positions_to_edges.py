###########################################################################################
#
# First Authors: Virgile de Saint Blanquat
# Inspired by mace-main/mace/data/neighborhood.py
#
###########################################################################################

import numpy as np
from matscipy.neighbours import neighbour_list
import torch


def positions_to_edges(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: torch.tensor,
    default_dtype: str,
    boxs_size: np.ndarray, #[x,y,z]
    pbc: bool = True
):
    """
    Positions, cutoff, box size ---> edge index, shifts, unit shifts.
    An edge is shifted if the two particles are connected through a box boundary, meaning the distance calculation needs to be adjusted.
    unit_shifts is a binary indicator (1 or 0) for each box edge.
    shifts are computed as unit_shifts * box_size.
    With the shift vector S, the distances D between atoms can be computed as: D = positions[j] - positions[i] + shifts.
    """

    sender, receiver, unit_shifts = neighbour_list(
        quantities="ijS", #‘i’ : first atom index ‘j’ : second atom index ‘d’ : absolute distance ‘D’ : distance vector ‘S’ : shift vector (number of cell boundaries crossed by the bond between atom i and j). With the shift vector S, the distances D between atoms can be computed from: D = a.positions[j]-a.positions[i]+S.dot(a.cell)
        pbc=np.array([pbc, pbc, pbc]), #pbc = False for no periodic bounary 
        cell=np.array([[boxs_size[0],0,0],[0,boxs_size[1],0],[0,0,boxs_size[2]]]),
        positions=positions,
        cutoff=np.float64(cutoff),
    )

    ##############################################################################
    #
    # len(unit_shifts) = len(sender) = len(receiver) -> number of neighbours
    # the shift is applyed on the receiver
    # positions[receiver] - (positions[sender] - shifts)
    #
    ##############################################################################

    # Eliminate self-edges that don't cross periodic boundaries
    true_self_edge = sender == receiver
    true_self_edge &= np.all(unit_shifts == 0, axis=1)
    keep_edge = ~true_self_edge

    # Note: after eliminating self-edges, it can be that no edges remain in this system
    # Note: self-edges can still remain if cutt_off > boxs_size

    sender = sender[keep_edge]
    receiver = receiver[keep_edge]
    unit_shifts = unit_shifts[keep_edge]

    edge_index = torch.tensor(np.stack((sender, receiver)), dtype=torch.long)  # [2, n_edges]'.T' (edge_index is naturaly transposed in torch_geometric.data so we don't need to apply it now)

    shifts = torch.tensor(np.dot(unit_shifts, np.array([[boxs_size[0],0,0],[0,boxs_size[1],0],[0,0,boxs_size[2]]])), dtype=getattr(torch, default_dtype))  # [n_edges, 3]

    return edge_index, shifts, unit_shifts