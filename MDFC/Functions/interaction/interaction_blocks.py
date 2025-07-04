###########################################################################################
#
# Script inspired by github.com/ACEsuit/mace/modules/blocks.py
#       First Authors: Ilyes Batatia, Gregor Simm
#       This program is distributed under the MIT License (see MIT.md)
#
# Adaptation for MPCP model and comments: Virgile de Saint Blanquat
###########################################################################################

from typing import List, Optional, Tuple
from e3nn import nn
from e3nn.util.jit import compile_mode
import torch.nn.functional
import torch
import cuequivariance_torch as cuet
import cuequivariance as cue
from ...tools.torch_scatter import scatter_sum
from ...tools.irreps_tools import reshape_irreps, tp_out_irreps, linear_out_irreps
from ...tools.config_tools import config_class_training


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    """Parent class for all interaction methods."""
    def __init__(
        self,
        node_attrs_irreps: cue.Irreps,
        node_feats_irreps: cue.Irreps,
        edge_attrs_irreps: cue.Irreps,
        edge_feats_irreps: cue.Irreps,
        target_irreps: cue.Irreps,
        hidden_irreps: cue.Irreps,
        config: config_class_training,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP
        self.config = config
        self._setup()
    


@compile_mode("script")
class ResidualElementDependentInteractionBlock(InteractionBlock):
    """
    Residual: adds node properties to the message output (skip connection).  
    Element-Dependent: message construction depend on node attributes (slower but potentially more accurate).
    """

    def _setup(self) -> None:

        ########################################################
        #
        # Residual + Element dependent -> 5 weights matrix and no perceptron
        #
        ########################################################

        # Linear neural network transformation of the node features (no dimension change)
        # Two matrix multiplications: node_feats * Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing information about the output shape of the intermediate tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )
    
        # 'Element-dependent' transformation
        # edge_feats * Weights + node_attrs[sender] * Weights = WEMB_Edge_feat
        self.Element_dependant_transformation = sum_tow_tensor_multiplied_by_Weights(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.mij_transformation.weight_numel,
        )

        # Verify that irreps_mid and target_irreps are compatible, and return irreps_mid
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()

        # Message Linear transformation for output
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Residual' transformation 
        # node_feats*node_attrs*Weights = sc
        self.Residual_transformation  =  cuet.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        sc = self.Residual_transformation(node_feats, node_attrs)
        weighted_node_feats = self.Weighted_node_feat_transformation(node_feats[sender])

        WEMB_Edge_feat = self.Element_dependant_transformation(node_attrs[sender], edge_feats)
        mji = self.mij_transformation(
            weighted_node_feats[sender], edge_attrs, WEMB_Edge_feat
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors
        return weighted_Message, sc  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticNonlinearInteractionBlock(InteractionBlock):
    """
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?)
    """
    def _setup(self) -> None:



        ########################################################
        #
        # Agnostic -> 3 weights matrix and a perceptron (Non linear)
        #
        ########################################################

        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, 
        )# If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
    
        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.mij_transformation.weight_numel],
            torch.nn.functional.silu,
        )

        # Verify that irreps_mid and target_irreps are compatible, and return irreps_mid
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()

        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )



        # Message Embeding with Node_atr and Weighted
        # weighted_Message*Node_atr*Weights = output
        self.output_transformation =  cuet.FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        WEMB_Edge_feats = self.perceptron(edge_feats)
        Weighted_node_feat = self.Weighted_node_feat_transformation(node_feats)

        mji = self.mij_transformation(
            Weighted_node_feat[sender], edge_attrs, WEMB_Edge_feats
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors
        output = self.output_transformation(weighted_Message, node_attrs)
        return output, None  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
    """
    Residual: add node property to the message output (sc) \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?)
    """
    def _setup(self) -> None:

        ########################################################
        #
        # Agnostic + Residual -> 3 weights matrix and a perceptron (Non linear)
        #
        ########################################################


        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )

        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Verify that irreps_mid and target_irreps are compatible, and return irreps_mid
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()

        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Residual' transformation 
        # node_feats*node_attrs*Weights = sc
        self.Residual_transformation  =  cuet.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        sc = self.Residual_transformation(node_feats, node_attrs)

        weighted_node_feats = self.Weighted_node_feat_transformation(node_feats)
        WEMB_Edge_feat = self.perceptron(edge_feats)
        mji = self.mij_transformation(
            weighted_node_feats[sender], edge_attrs, WEMB_Edge_feat
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors
        return weighted_Message, sc  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    """
    Real: ? to test \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?)
    """
    def _setup(self) -> None:



        ########################################################
        #
        # Agnostic -> 3 weights matrix and a perceptron (Non linear)
        # real : self.irreps_out = self.target_irreps

        #
        ########################################################

        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )
    
        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.mij_transformation.weight_numel],
            torch.nn.functional.silu,
        )

        # 'real'
        self.irreps_out = self.target_irreps


        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )



        # Message Embeding with Node_atr and Weighted
        # weighted_Message*Node_atr*Weights = output
        self.output_transformation =  cuet.FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

        #'real'
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.group)


    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        WEMB_Edge_feats = self.perceptron(edge_feats)
        Weighted_node_feat = self.Weighted_node_feat_transformation(node_feats)

        mji = self.mij_transformation(
            Weighted_node_feat[sender], edge_attrs, WEMB_Edge_feats
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors
        output = self.output_transformation(weighted_Message, node_attrs)
        return self.reshape(output), None  # [n_nodes, irreps]



@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    """
    Residual: add node property to the message output (sc) \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?) \n
    Real : ?
    """
    def _setup(self) -> None:

        ########################################################
        #
        # Agnostic + Residual -> 3 weights matrix and a perceptron (Non linear)
        # Real : self.irreps_out = self.target_irreps
        #
        ########################################################


        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # 3 Matrix Tensor Product With no Learning associated (ChannelWiseTensorProduct is faster than others methods)
        # weighted_node_feats[sender]*edge_attrs*(')WEMB_Edge_feat(') = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )

        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.mij_transformation.weight_numel],
            torch.nn.functional.silu,
        )

        # 'real'
        self.irreps_out = self.target_irreps

        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Residual' transformation 
        # node_feats*node_attrs*Weights = sc
        self.Residual_transformation  =  cuet.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.config.layout)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        sc = self.Residual_transformation(node_feats, node_attrs)

        weighted_node_feats = self.Weighted_node_feat_transformation(node_feats)
        WEMB_Edge_feat = self.perceptron(edge_feats)

        mji = self.mij_transformation(
            weighted_node_feats[sender], edge_attrs, WEMB_Edge_feat
        )  # [n_edges, irreps]

        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]

        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors

        return self.reshape(weighted_Message), sc  # [n_nodes, irreps]
    

@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    """
    Real: ? to test \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?) \n
    Density: density normalisation
    """
    def _setup(self) -> None:



        ########################################################
        #
        # Agnostic -> 3 weights matrix and a perceptron (Non linear)
        # real : self.irreps_out = self.target_irreps
        #
        ########################################################

        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )
    
        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.mij_transformation.weight_numel],
            torch.nn.functional.silu,
        )

        # 'real'
        self.irreps_out = self.target_irreps


        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )



        # Message Embeding with Node_atr and Weighted
        # weighted_Message*Node_atr*Weights = output
        self.output_transformation =  cuet.FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

        #'real'
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.group)

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim] + [1],
            torch.nn.functional.silu,
        )


    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        WEMB_Edge_feats = self.perceptron(edge_feats)
        Weighted_node_feat = self.Weighted_node_feat_transformation(node_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)

        mji = self.mij_transformation(
            Weighted_node_feat[sender], edge_attrs, WEMB_Edge_feats
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / (density + 1)
        output = self.output_transformation(weighted_Message, node_attrs)
        return self.reshape(output), None  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    """
    Residual: add node property to the message output (sc) \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?) \n
    Density: density normalisation \n
    Real : ?
    """
    def _setup(self) -> None:

        ########################################################
        #
        # Agnostic + Residual -> 3 weights matrix and a perceptron (Non linear)
        # Real : self.irreps_out = self.target_irreps
        #
        ########################################################


        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = weighted_node_feats
        self.Weighted_node_feat_transformation = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )

        # 'Agnostic' Transformation
        # Perceptron to embed Edge_feat
        input_dim = self.edge_feats_irreps.num_irreps
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # 'real'
        self.irreps_out = self.target_irreps

        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Residual' transformation 
        # node_feats*node_attrs*Weights = sc
        self.Residual_transformation  =  cuet.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )


        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim] + [1],
            torch.nn.functional.silu,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.config.layout)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        sc = self.Residual_transformation(node_feats, node_attrs)

        weighted_node_feats = self.Weighted_node_feat_transformation(node_feats)
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        WEMB_Edge_feat = self.perceptron(edge_feats)
        mji = self.mij_transformation(
            weighted_node_feats[sender], edge_attrs, WEMB_Edge_feat
        )  # [n_edges, irreps]
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / (density + 1)
        return self.reshape(weighted_Message), sc  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    """
    Residual: add node property to the message output (sc) \n
    Agnostic: weights don't depend on the node_attrs (faster but less accurate model?) \n
    Att: augmented Edge_feat with Node_feat Concatenates \n
    Real : ?
    """
    def _setup(self) -> None:

        ########################################################
        #
        # Agnostic + Residual -> 3 weights matrix and a perceptron (Non linear)
        # Real : self.irreps_out = self.target_irreps
        #
        ########################################################

        self.node_feats_down_irreps = cue.Irreps("O3", "64x0e")

        # Linear NN transformation of the nodes features (no dimension change)
        # 2 matrix multiplication node_feats*Weights = node_feats_up (weighted_node_feats)
        self.Weighted_node_feat_transformation_up = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Att' Transformation
        # Linear NN transformation of the nodes features (dimension change!)
        # 2 matrix multiplication node_feats*Weights = node_feats_down
        self.Weighted_node_feat_transformation_down = cuet.Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )
        
        # irreps_mid: Irreps object containing the information of the output shape of the intermediate Tensors mij
        irreps_mid = tp_out_irreps(
            self.node_feats_irreps, 
            self.edge_attrs_irreps, 
            self.target_irreps
        )

        # Three-way tensor product with no learning involved (ChannelWiseTensorProduct is faster than other methods)
        # weighted_node_feats[sender] * edge_attrs * WEMB_Edge_feat = mji
        self.mij_transformation = cuet.ChannelWiseTensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            layout=self.config.layout,
            shared_weights=False,
            internal_weights=False,
            use_fallback=True, # If False: 'RuntimeError: can't query kernel attributes'. Check why this happens and whether it needs to be fixed for False. todo: Get rid of this?
        )

        # 'Agnostic' + 'Att' Transformation
        # Perceptron to embed Edge_feat + node_feats
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )
        self.perceptron = nn.FullyConnectedNet(
            [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # 'real'
        self.irreps_out = self.target_irreps

        # Message Linear transformation 
        # Message*Weights = weighted_Message
        self.weighted_Message_Transformation = cuet.Linear(
            irreps_mid,
            self.irreps_out,
            layout=self.config.layout,
            shared_weights=True,
            use_fallback=True,
        )

        # 'Residual' transformation 
        # node_feats*node_attrs*Weights = sc
        self.Residual_transformation  =  cuet.FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            layout=self.config.layout,
            shared_weights=True,
            internal_weights=True,
            use_fallback=True,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.config.layout)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]

        sc = self.Residual_transformation(node_feats, node_attrs)        

        node_feats_up = self.Weighted_node_feat_transformation_up(node_feats)
        node_feats_down = self.Weighted_node_feat_transformation_down(node_feats)

        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )

        WEMB_Edge_feat = self.perceptron(augmented_edge_feats)

        mji = self.mij_transformation(
            node_feats_up[sender], edge_attrs, WEMB_Edge_feat
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        weighted_Message = self.weighted_Message_Transformation(message) / self.config.avg_num_neighbors
        return self.reshape(weighted_Message), sc  # [n_nodes, irreps]