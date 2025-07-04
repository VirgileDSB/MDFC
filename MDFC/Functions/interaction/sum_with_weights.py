import torch
from e3nn.util.jit import compile_mode
import numpy as np

@compile_mode("script")
class sum_two_tensor_multiplied_by_Weights(torch.nn.Module): # Old TensorProductWeightsBlock in mace_main/mace/modules/blocks.py
    """Neural Network step class, used to sum output_message and Lat_feat_space with learnable weights multiplication \n
    ---> init: create weight matrix \n
    ---> forward: sum: edge_feats[be], node_attrs[ba], weights[aek] -> new_LFS[bk] \n
    ---> repr: return weight values
    """
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )