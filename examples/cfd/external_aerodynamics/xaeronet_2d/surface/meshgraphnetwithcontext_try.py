# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code defines a distributed training pipeline for training MeshGraphNet at scale,
which operates on partitioned graph data for the naca 0012 dataset. It includes
loading partitioned graphs from .bin files, normalizing node and edge features using
precomputed statistics, and training the model in parallel using DistributedDataParallel
across multiple GPUs. The training loop involves computing predictions for each graph
partition, calculating loss, and updating model parameters using mixed precision.
Periodic checkpointing is performed to save the model, optimizer state, and training
progress. Validation is also conducted every few epochs, where predictions are compared
against ground truth values, and results are saved as point clouds. The code logs training
and validation metrics to TensorBoard and optionally integrates with Weights and Biases for
experiment tracking.
"""

import os
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch import Tensor

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
from dataclasses import dataclass
from itertools import chain
from typing import Callable, List, Tuple, Union
import torch.nn.functional as F
import modulus  # noqa: F401 for docs
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module
from modulus.models.meshgraphnet import MeshGraphNet 
from modulus.models.gnn_layers.utils import CuGraphCSC, aggregate_and_concat

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

class MeshGraphNetWithContext(MeshGraphNet):
    def __init__(self,         
        input_dim_nodes,
        input_dim_edges,
        output_dim,
        context_dim,  # Add context dimension
        processor_size=10,
        mlp_activation_fn="relu",
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        hidden_dim_processor=128,
        hidden_dim_node_encoder=128,
        num_layers_node_encoder=2,
        hidden_dim_edge_encoder=128,
        num_layers_edge_encoder=2,
        hidden_dim_node_decoder=256,
        num_layers_node_decoder=2,
        aggregation="sum",
        do_concat_trick=False,
        num_processor_checkpoint_segments=0,
        recompute_activation=False,
        dropout_rate=0.3  # Add dropout rate
    ):
        super(MeshGraphNetWithContext, self).__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            recompute_activation=recompute_activation,
        )
        self.context_dim = context_dim
        self.dropout_rate = dropout_rate  # Store dropout rate

        # Initialize context encoder
        self.context_encoder = MeshGraphMLP(
            context_dim,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_processor,
            hidden_layers=2,
            activation_fn=get_activation(mlp_activation_fn),
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # Override the processor to include context
        self.processor = MeshGraphNetProcessor(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            input_dim_context=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            aggregation=aggregation,
            norm_type="LayerNorm",
            activation_fn=get_activation(mlp_activation_fn),
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=False,
        )

        # Override the node decoder to include context
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor + hidden_dim_processor,  # processor output + context
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=get_activation(mlp_activation_fn),
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def forward(self, node_features: Tensor, edge_features: Tensor, graph: DGLGraph, context: Tensor) -> Tensor:
        # Ensure edge_features has the correct shape
        if edge_features.shape[1] != self.edge_encoder.model[0].in_features:
            raise ValueError(f"Expected edge_features with {self.edge_encoder.model[0].in_features} features, but got {edge_features.shape[1]}")

        # Convert input tensors to the same dtype as model parameters
        node_features = node_features.to(self.node_encoder.model[0].weight.dtype)
        edge_features = edge_features.to(self.edge_encoder.model[0].weight.dtype)
        context = context.to(self.context_encoder.model[0].weight.dtype)

        # Encode node and edge features with dropout
        edge_features = F.dropout(self.edge_encoder(edge_features), p=self.dropout_rate, training=self.training)
        node_features = F.dropout(self.node_encoder(node_features), p=self.dropout_rate, training=self.training)

        # Encode context
        context_encoded = self.context_encoder(context)
        context_node_expanded = context_encoded.unsqueeze(0).expand(node_features.size(0), -1)
        context_edge_expanded = context_encoded.unsqueeze(0).expand(edge_features.size(0), -1)

        # Process node and edge features with context
        node_features = self.processor(node_features, edge_features, graph, context_node_expanded, context_edge_expanded)

        # Expand context and concatenate with node features before decoding
        node_features = torch.cat([node_features, context_node_expanded], dim=1)

        # Decode node features with dropout
        node_features = F.dropout(self.node_decoder(node_features), p=self.dropout_rate, training=self.training)
        return node_features
    
class MeshGraphNetProcessor(nn.Module):
    """MeshGraphNet processor block with context integration"""

    def __init__(
        self,
        processor_size: int = 15,
        input_dim_node: int = 128,
        input_dim_edge: int = 128,
        input_dim_context: int = 128,
        num_layers_node: int = 2,
        num_layers_edge: int = 2,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.ReLU(),
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.processor_size = processor_size
        self.num_processor_checkpoint_segments = num_processor_checkpoint_segments
        self.checkpoint_offloading = (
            checkpoint_offloading if (num_processor_checkpoint_segments > 0) else False
        )
        self.dropout_rate = dropout_rate

        self.input_dim_node = input_dim_node
        self.input_dim_edge = input_dim_edge
        self.input_dim_context = input_dim_context

        # Define processor layers with context integration
        self.processor_layers = nn.ModuleList()
        for _ in range(processor_size):
            edge_block = MeshEdgeBlock(
                input_dim_node,
                input_dim_edge,
                input_dim_edge,
                input_dim_edge,
                num_layers_edge,
                activation_fn,
                norm_type,
                do_concat_trick,
                False,
            )
            node_block = MeshNodeBlock(
                aggregation,
                input_dim_node + input_dim_context,  # Adjusted to include context
                input_dim_edge,
                input_dim_node,
                input_dim_node,
                num_layers_node,
                activation_fn,
                norm_type,
                False,
            )
            self.processor_layers.append(nn.ModuleList([edge_block, node_block]))

        self.num_processor_layers = len(self.processor_layers)
        self.set_checkpoint_segments(self.num_processor_checkpoint_segments)
        self.set_checkpoint_offload_ctx(self.checkpoint_offloading)

    def set_checkpoint_offload_ctx(self, enabled: bool):
        """Set CPU offloading context for checkpoints."""
        if enabled:
            self.checkpoint_offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
        else:
            self.checkpoint_offload_ctx = nullcontext()

    def set_checkpoint_segments(self, checkpoint_segments: int):
        """Set the number of checkpoint segments."""
        if checkpoint_segments > 0:
            if self.num_processor_layers % checkpoint_segments != 0:
                raise ValueError("Processor layers must be a multiple of checkpoint_segments")
            segment_size = self.num_processor_layers // checkpoint_segments
            self.checkpoint_segments = []
            for i in range(0, self.num_processor_layers, segment_size):
                self.checkpoint_segments.append((i, i + segment_size))
            self.checkpoint_fn = set_checkpoint_fn(True)
        else:
            self.checkpoint_fn = set_checkpoint_fn(False)
            self.checkpoint_segments = [(0, self.num_processor_layers)]

    def run_function(
        self, segment_start: int, segment_end: int, context_node: Tensor, context_edge: Tensor
    ) -> Callable[[Tensor, Tensor, Union[DGLGraph, List[DGLGraph]]], Tuple[Tensor, Tensor]]:
        """Custom forward function with context integration."""
        segment = self.processor_layers[segment_start:segment_end]

        def custom_forward(
            node_features: Tensor,
            edge_features: Tensor,
            graph: Union[DGLGraph, List[DGLGraph]],
        ) -> Tuple[Tensor, Tensor]:
            if edge_features.shape[0] != graph.num_edges():
                raise ValueError(
                    f"Edge features size {edge_features.shape[0]} does not match graph edges {graph.num_edges()}"
                )
            for edge_block, node_block in segment:
                # Edge update
                edge_features_new, _ = edge_block(edge_features, node_features, graph)
                edge_features = F.dropout(edge_features_new, p=self.dropout_rate, training=self.training)
                print(f"edge_features shape after edge_block: {edge_features.shape}")
                # Node update with context, preserving edge_features
                node_input = torch.cat([node_features, context_node], dim=1)
                print(f"node_input shape: {node_input.shape}")
                edge_features_out, node_features = node_block(node_input, edge_features, graph)
                print(f"edge_features shape after node_block: {edge_features.shape}")
                print(f"node_features shape after node_block: {node_features.shape}")
                # If node_block reduces edge_features, revert to edge_block output
                if edge_features_out.shape[0] != graph.num_edges():
                    print(f"Warning: node_block reduced edge_features from {edge_features.shape[0]} to {edge_features_out.shape[0]}. Using edge_block output.")
                else:
                    edge_features = edge_features_out
                node_features = torch.cat([node_features, context_node], dim=1)[:, :self.input_dim_node]
                print(f"node_features shape final: {node_features.shape}")
            return edge_features, node_features

        return custom_forward

    @torch.jit.unused
    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
        context_node: Tensor,
        context_edge: Tensor,
    ) -> Tensor:
        """Process features with context-augmented message passing."""
        with self.checkpoint_offload_ctx:
            for segment_start, segment_end in self.checkpoint_segments:
                edge_features, node_features = self.checkpoint_fn(
                    self.run_function(segment_start, segment_end, context_node, context_edge),
                    node_features,
                    edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
        return node_features
    
class MeshNodeBlock(nn.Module):
    """Node block used e.g. in GraphCast or MeshGraphNet
    operating on a latent space represented by a mesh.

    Parameters
    ----------
    aggregation : str, optional
        Aggregation method (sum, mean) , by default "sum"
    input_dim_nodes : int, optional
        Input dimensionality of the node features, by default 512
    input_dim_edges : int, optional
        Input dimensionality of the edge features, by default 512
    output_dim : int, optional
        Output dimensionality of the node features, by default 512
    hidden_dim : int, optional
        Number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        Number of neurons in each hidden layer, by default 1
    activation_fn : nn.Module, optional
       Type of activation function, by default nn.SiLU()
    norm_type : str, optional
        Normalization type ["TELayerNorm", "LayerNorm"].
        Use "TELayerNorm" for optimal performance. By default "LayerNorm".
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        aggregation: str = "sum",
        input_dim_nodes: int = 512,
        input_dim_edges: int = 512,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation

        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim_nodes + input_dim_edges,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation,
        )

    @torch.jit.ignore()
    def forward(
        self,
        efeat: Tensor,
        nfeat: Tensor,
        graph: Union[DGLGraph, CuGraphCSC],
    ) -> Tuple[Tensor, Tensor]:
        # update edge features
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        # update node features + residual connection
        nfeat_new = self.node_mlp(cat_feat) + nfeat
        return efeat, nfeat_new