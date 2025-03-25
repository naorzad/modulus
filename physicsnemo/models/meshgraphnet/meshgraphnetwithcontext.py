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

try:
    import dgl  # noqa: F401 for docs
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
from typing import List, Tuple, Union
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple
from modulus.models.meshgraphnet import MeshGraphNet, MeshGraphNetProcessor
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.utils import CuGraphCSC
from modulus.models.layers import get_activation

class MeshGraphNetWithContext(MeshGraphNet):
    """MeshGraphNet with enhanced context integration for flow parameter influence.

    This class extends MeshGraphNet to incorporate a context vector (e.g., flow parameters)
    into the processor's edge and node updates, amplifying its impact on predictions.
    It reuses the base MeshGraphNet's encoders and processor structure, adding a context
    encoder and modifying the forward pass to include context in message passing.

    Parameters
    ----------
    input_dim_nodes : int
        Number of node features (e.g., coordinates, normals, Fourier terms).
    input_dim_edges : int
        Number of edge features (e.g., relative displacements).
    output_dim : int
        Number of output features (e.g., pressure + shear stress).
    context_dim : int
        Number of context features (e.g., Mach number, Reynolds number).
    processor_size : int, optional
        Number of message-passing layers, default is 15.
    mlp_activation_fn : Union[str, List[str]], optional
        Activation function, default is "relu".
    num_layers_node_processor : int, optional
        Layers in node processor MLPs, default is 2.
    num_layers_edge_processor : int, optional
        Layers in edge processor MLPs, default is 2.
    hidden_dim_processor : int, optional
        Hidden size for processor MLPs, default is 128.
    hidden_dim_node_encoder : int, optional
        Hidden size for node encoder, default is 128.
    num_layers_node_encoder : Union[int, None], optional
        Layers in node encoder, default is 2.
    hidden_dim_edge_encoder : int, optional
        Hidden size for edge encoder, default is 128.
    num_layers_edge_encoder : Union[int, None], optional
        Layers in edge encoder, default is 2.
    hidden_dim_node_decoder : int, optional
        Hidden size for node decoder, default is 128.
    num_layers_node_decoder : Union[int, None], optional
        Layers in node decoder, default is 2.
    aggregation : str, optional
        Message aggregation type, default is "sum".
    do_concat_trick : bool, optional
        Use concatenation trick, default is False.
    num_processor_checkpoint_segments : int, optional
        Number of checkpoint segments, default is 0.
    recompute_activation : bool, optional
        Recompute activations, default is False.
    dropout_rate : float, optional
        Dropout rate for regularization, default is 0.3.

    Example
    -------
    >>> from meshgraphnet_with_context import MeshGraphNetWithContext
    >>> model = MeshGraphNetWithContext(input_dim_nodes=16, input_dim_edges=3, output_dim=3, context_dim=4)
    >>> graph = dgl.rand_graph(10, 5)
    >>> node_features = torch.randn(10, 16)
    >>> edge_features = torch.randn(5, 3)
    >>> context = torch.randn(4)
    >>> output = model(node_features, edge_features, graph, context)
    >>> output.shape
    torch.Size([10, 3])
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        context_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: Union[str, list[str]] = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: str = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        recompute_activation: bool = False,
        dropout_rate: float = 0.3,
    ):
        # Initialize the base MeshGraphNet with inherited parameters
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
        self.dropout_rate = dropout_rate
        activation_fn = get_activation(mlp_activation_fn)

        # Add a context encoder to transform context into hidden_dim_processor space
        self.context_encoder = MeshGraphMLP(
            in_features=context_dim,
            out_features=hidden_dim_processor,
            hidden_dim=hidden_dim_processor,
            hidden_layers=2,
            activation_fn=activation_fn,
            norm_type="LayerNorm",
            recompute_activation=recompute_activation,
        )

        # Override the processor to include context in each layer
        # We'll reuse the base processor's structure but adjust inputs in forward
        # No need to redefine processor here; inherit it and modify behavior in forward

        # Override the node decoder to include context
        self.node_decoder = MeshGraphMLP(
            in_features=hidden_dim_processor + hidden_dim_processor,  # processor output + context
            out_features=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

    def forward(
            self,
            node_features: Tensor,
            edge_features: Tensor,
            graph: Union[DGLGraph, List[DGLGraph], CuGraphCSC],
            context: Tensor,
        ) -> Tensor:
            """Forward pass with context integrated into the processor.

            Args:
                node_features: Input node features, shape (N, input_dim_nodes).
                edge_features: Input edge features, shape (E, input_dim_edges).
                graph: Graph structure (DGLGraph, list of DGLGraphs, or CuGraphCSC).
                context: Context vector (e.g., flow parameters), shape (context_dim,).

            Returns:
                Predictions, shape (N, output_dim).
            """
            # Validate edge features
            if edge_features.shape[1] != self.edge_encoder.model[0].in_features:
                raise ValueError(
                    f"Expected edge_features with {self.edge_encoder.model[0].in_features} features, "
                    f"but got {edge_features.shape[1]}"
                )

            # Ensure consistent data types
            node_features = node_features.to(self.node_encoder.model[0].weight.dtype)
            edge_features = edge_features.to(self.edge_encoder.model[0].weight.dtype)
            context = context.to(self.context_encoder.model[0].weight.dtype)

            # Encode node and edge features using inherited encoders
            node_features = F.dropout(
                self.node_encoder(node_features), p=self.dropout_rate, training=self.training
            )
            edge_features = F.dropout(
                self.edge_encoder(edge_features), p=self.dropout_rate, training=self.training
            )

            # Encode context and expand to match node and edge counts
            context_encoded = self.context_encoder(context)  # Shape: (hidden_dim_processor,)
            context_node_expanded = context_encoded.expand(node_features.size(0), -1)  # (N, hidden_dim_processor)
            context_edge_expanded = context_encoded.expand(edge_features.size(0), -1)  # (E, hidden_dim_processor)

            # Augment node features with context before processing
            node_features_with_context = torch.cat([node_features, context_node_expanded], dim=1)  # (N, 256)

            # Process with inherited processor, adjusting edge inputs dynamically
            for segment_start, segment_end in self.processor.checkpoint_segments:
                segment = self.processor.processor_layers[segment_start:segment_end]
                for module in segment:
                    if isinstance(module, MeshEdgeBlock):
                        # Augment edge features with context for edge updates
                        edge_input = torch.cat(
                            [
                                edge_features,
                                node_features[graph.edges()[0]],  # Source nodes
                                node_features[graph.edges()[1]],  # Destination nodes
                                context_edge_expanded,
                            ],
                            dim=1,
                        )  # (E, 128 + 128 + 128 + 128 = 512)
                        edge_features = module(edge_input)
                    else:  # MeshNodeBlock
                        # Use augmented node features for node updates
                        edge_features, node_features = module(
                            node_features_with_context, edge_features, graph
                        )
                        # Re-augment node features after update to maintain context
                        node_features_with_context = torch.cat([node_features, context_node_expanded], dim=1)

            # Decode with context
            output = self.node_decoder(node_features_with_context)
            return F.dropout(output, p=self.dropout_rate, training=self.training)