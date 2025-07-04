from typing import List, Optional, Tuple
import torch



def compute_forces(
    Effective_energy: torch.Tensor, 
    positions: torch.Tensor, 
    training: bool
) -> torch.Tensor:
    """Compute forces from the effective energy using autograd."""
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(Effective_energy)]
    gradient = torch.autograd.grad(
        outputs=[Effective_energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[0]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute edges vectors and edges lengths from positions."""
    sender = edge_index[0]
    receiver = edge_index[1]

    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths