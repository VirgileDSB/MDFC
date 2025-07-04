###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/modules/blocks.py
#       First Authors: Ilyes Batatia, Gregor Simm, David Kovacs
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

from ..tools.config_tools import config_load_model
from typing import Dict, Optional, Type
import logging
logger = logging.getLogger("Main_Logger")

import cuequivariance as cue
import torch
from e3nn import o3
from e3nn.util.jit import compile_mode


from ..Functions.Product.symmetrisation import EquivariantProductBasisBlock

from ..tools.calculators import get_edge_vectors_and_lengths, compute_forces
from ..tools.torch_scatter import scatter_sum
from ..tools.Wrappers import Linear_wrapper, Non_Linear_wrapper, ScaleShiftBlock
from ..tools.printing import update_progress

from ..Functions.embedding_fcts.embedding_fct import RadialEmbeddingBlock
from ..Functions.interaction.interaction_blocks import InteractionBlock


@compile_mode("script")
class MPCP(torch.nn.Module):
    """
    Model
    """

    def __init__(
        self,
        config: config_load_model,
        interaction_cls_first: Type[InteractionBlock],
        interaction_cls: Type[InteractionBlock],
    ):
        super().__init__()

        #For dynamic printing
        print(f"\033[34m Starting building Model \033[0m")

        #Number of functional blocks to build
        tot_to_build = 3 * (config.num_interactions)
        
        update_progress(0.0/tot_to_build)


        ##################### SETUP #####################

        # register_buffer because thoses 2 are hypers parameters so we want to save theme for model usage
        self.register_buffer("cutt_off", torch.tensor(config.cutt_off, dtype=torch.get_default_dtype()))
        self.register_buffer("num_interactions", torch.tensor(config.num_interactions, dtype=torch.int64))

        # config.correlation is juste one element -> all correlations are be the same -> expend to a num_interactions long list with only this element
        if type(config.correlation) == int:
            config.correlation = [config.correlation] * config.num_interactions        
        elif len(config.correlation) == 1:
            config.correlation = config.correlation * config.num_interactions
        # config.correlation is is setup as a list -> it need to be a num_interactions long list
        elif len(config.correlation) != config.num_interactions:
            logger.error(f"len(config.correlation) != config.num_interactions")

        ##################### Embedding #####################

        num_features = config.num_channels
        node_attrs_irreps = cue.Irreps("O3", [(config.num_elements, (0, 1))]) # Irreps for One-Hot encoding
        node_feats_irreps = cue.Irreps("O3", [(num_features, (0, 1))]) # Irreps for Lattent Feature Space
        edge_feats_irreps = cue.Irreps("O3", f"{config.num_radial_basis}x0e") # Irreps for edge Bessel représentation
        sh_irreps = o3.Irreps.spherical_harmonics(config.max_ell) # Irreps for edge Spherical Harmonics angular représentation (e3nn type for o3.SphericalHarmonics)
        interaction_irreps = cue.Irreps("O3", (sh_irreps * num_features).sort()[0].simplify()) # Irreps for interaction Block calculation/output

        # node_embedding setup: One-Hot encoding -> Lattent Feature Space
        self.node_embedding = Linear_wrapper(
            irreps_in = node_attrs_irreps,
            irreps_out = node_feats_irreps,
            config = config,
        )


        # radial_embedding: Scalar distance -> Bessels distances
        self.radial_embedding = RadialEmbeddingBlock(
            r_max = config.cutt_off,
            num_bessel = config.num_radial_basis,
            num_polynomial_cutoff = config.num_polynomial_cutoff,
        )

        update_progress(1.0/tot_to_build)

        # Angular Embedding: 3D positions -> Sphericals Harmonics
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, 
            normalize=True, # No radial information needed so we use the normalized version
            normalization="component"
        )

        sh_irreps = cue.Irreps("O3", sh_irreps) # convert to cuequivariance type for the rest

        ##################### Interactions 1st layer #####################

        # Interactions Block for the 1st message passing process
        inter = interaction_cls_first(
            node_attrs_irreps = node_attrs_irreps, # todo: rename to one_hot_irreps for better understanding
            node_feats_irreps = node_feats_irreps, # todo: rename to latent_irreps for better understanding
            edge_attrs_irreps = sh_irreps, # edge_attrs_irreps is the spherical harmonics (todo: rename to sh_irreps for better understanding)
            edge_feats_irreps = edge_feats_irreps, # todo: rename to radial_irreps for better understanding
            target_irreps = interaction_irreps, # target_irreps is output irreps of the interaction block
            hidden_irreps = config.hidden_irreps, # hidden_irreps is the irreps of the hidden layer
            radial_MLP = config.radial_MLP, # Radial MLP is optional for 'Agnostic' Transformation
            config = config
        )

        update_progress(2.0/tot_to_build)

        # Create the interaction block ModuleList
        self.interactions = torch.nn.ModuleList([inter])

        # Check if the first interaction block is a residual block
        use_sc_first = False
        if "Residual" in str(interaction_cls_first):
            use_sc_first = True


        ##################### Product 1st layer #####################

        # product Block for the 1st message passing process
        prod = EquivariantProductBasisBlock(
            node_feats_irreps=inter.target_irreps,
            target_irreps=config.hidden_irreps,
            config=config,
            use_sc=use_sc_first,
            correlation=config.correlation[0],
        )

        # Create the product block ModuleList
        self.products = torch.nn.ModuleList([prod])

        update_progress(3.0/tot_to_build)


        ##################### Readout 1st layer #####################

        # Create the Readout block ModuleList
        self.readouts = torch.nn.ModuleList()

        # Readout Block for the 1st message passing process
        self.readouts.append(Linear_wrapper(config.hidden_irreps, cue.Irreps("O3", f"0e"), config))



        ##################### Others Layers #####################

        for i in range(config.num_interactions - 1): # The first layer is already set so loop for all others

            if i == config.num_interactions - 2: # if last layer
                hidden_irreps_out = cue.Irreps("O3", str(config.hidden_irreps[0]))  # Select only scalars for last layer

            else: # Middle layers (if some)
                hidden_irreps_out = config.hidden_irreps

            # Create a interaction block and add it to the ModuleList (check interaction_cls_first for more infos)
            inter = interaction_cls(
                node_attrs_irreps=node_attrs_irreps,
                node_feats_irreps=config.hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                radial_MLP=config.radial_MLP,
                config = config
            )
            self.interactions.append(inter)

            update_progress(((i+1)*3 + 1)/tot_to_build)

            # Create the product block and add it to the ModuleList
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps, #todo: change that name
                target_irreps=hidden_irreps_out,
                config=config,
                use_sc=True, # tocheck: alwayse true??
                correlation = config.correlation[i + 1],
            )
            self.products.append(prod)

            update_progress(((i+1)*3 + 2)/tot_to_build)

            # Create the readout block and add it to the ModuleList
            if i == config.num_interactions - 2: # if last layer redout is non-linear
                self.readouts.append(
                    Non_Linear_wrapper(
                        irreps_in = hidden_irreps_out,
                        MLP_irreps = config.MLP_irreps,
                        config = config,
                        irrep_out = cue.Irreps("O3", f"0e"),
                    )
                )
            else:
                self.readouts.append(
                    Linear_wrapper(
                        config.hidden_irreps, cue.Irreps("O3", f"0e"), config=config
                    )
                )

            update_progress(((i+1)*3 + 3)/tot_to_build)
        
        # todo: implement squale/shift
        # self.scale_shift = ScaleShiftBlock(shift=0)

        print()
        print()





    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        

        ##################### SETUP #####################

        data["positions"].requires_grad_(True)
        data["x"].requires_grad_(True)
        num_atoms_arange = torch.arange(data["positions"].shape[0])


        ##################### Embedding #####################

        node_feats = self.node_embedding(data["x"]) # data["x"] --> node_attrs
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"], # Rename 'shifts' to avoid confusion with the ScaleShiftBlock
        )

        # We apply edge embedding
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)


        ##################### Message passing Layers #####################

        node_energies_list = []

        for interaction, product, readout in zip(self.interactions, self.products, self.readouts):

            # Individual interaction of each edges
            interaction_feats, sc = interaction(
                node_attrs=data["x"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )
            
            # Product of all individual interactions 
            node_feats = product(
                node_feats=interaction_feats, 
                sc=sc, 
                node_attrs=data["x"]
            )

            # Readout of the energy of each node
            node_energies = readout(node_feats)[num_atoms_arange]

            # Save the node energies for each interaction step
            node_energies_list.append(node_energies)


        #################### Output #####################

        # node_energies_list should have shape [num_interactions, n_nodes]

        # Sum over interactions steps to have total nodes energy
        tot_nodes_energy = torch.sum(torch.stack(node_energies_list, dim=0), dim=0)  # tot_nodes_energy should be [n_nodes]
        
        # for testing purpose, to deleet later
        if tot_nodes_energy.squeeze().size() != data["batch"].size():
            raise ValueError(f"tot_nodes_energy.size() != data['batch'].size() : {tot_nodes_energy.squeeze().size()} != {data['batch'].size()}")


        # Apply the scale and shift bock (not implemented yet)
        # tot_nodes_energy = self.scale_shift(tot_nodes_energy)


        # Sum over nodes in graph to have a file total energy 
        energy_per_snapshoot = scatter_sum(src=tot_nodes_energy.squeeze(), index=data["batch"], dim=0)  # energy_per_graph should be [num_files]

        # Compute forces
        forces = compute_forces(
            energy=energy_per_snapshoot,
            positions=data["positions"],
            training=training,
        )

        output = {
            "snapshots_energy": energy_per_snapshoot,
            "node_energy": tot_nodes_energy,
            "forces": forces,
        }

        return output