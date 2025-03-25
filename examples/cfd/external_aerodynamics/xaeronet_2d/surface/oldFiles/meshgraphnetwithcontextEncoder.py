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

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Callable, Tuple, List
from itertools import chain
from contextlib import nullcontext

try:
    import dgl
    from dgl import DGLGraph
except ImportError:
    raise ImportError(
        "MeshGraphNet requires the DGL library. Install the desired CUDA version at: "
        "https://www.dgl.ai/pages/start.html"
    )
from dataclasses import dataclass
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.utils import CuGraphCSC, set_checkpoint_fn
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module
from modulus.models.meshgraphnet import MeshGraphNet 

def get_activation(name: Union[str, List[str]]) -> nn.Module:
    if isinstance(name, str):
        if name.lower() == "relu":
            return nn.ReLU()
        elif name.lower() == "silu":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {name}")
    else:
        raise ValueError("Activation must be a string")

def aggregate_and_concat(
    efeat: torch.Tensor,
    nfeat: torch.Tensor,
    graph: DGLGraph,
    aggregation: str = "sum"
) -> torch.Tensor:
    """Aggregates edge features per node and concatenates with node features."""
    with graph.local_scope():
        graph.ndata['n'] = nfeat
        graph.edata['e'] = efeat
        if aggregation == "sum":
            graph.update_all(dgl.function.copy_e('e', 'm'), dgl.function.sum('m', 'e_agg'))
        elif aggregation == "mean":
            graph.update_all(dgl.function.copy_e('e', 'm'), dgl.function.mean('m', 'e_agg'))
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        return torch.cat([graph.ndata['n'], graph.ndata['e_agg']], dim=-1)

class MeshGraphMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False
    ):
        super().__init__()
        self.recompute_activation = recompute_activation
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim) if norm_type == "LayerNorm" else nn.Identity())
        layers.append(activation_fn)
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim) if norm_type == "LayerNorm" else nn.Identity())
            layers.append(activation_fn)
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class MeshGraphEdgeMLPConcatWithContext(nn.Module):
    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        flow_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False
    ):
        super().__init__()
        # Total input dimension includes edge features, src/dst node features, and src/dst flow features
        input_dim = efeat_dim + src_dim + dst_dim + 2 * flow_dim
        self.model = MeshGraphMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation
        )

    def forward(
        self,
        efeat: torch.Tensor,
        nfeat: torch.Tensor,
        flow_features: torch.Tensor,
        graph: DGLGraph
    ) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['n'] = nfeat
            graph.ndata['f'] = flow_features
            graph.edata['e'] = efeat
            graph.apply_edges(
                lambda edges: {
                    'cat': torch.cat(
                        [edges.data['e'], edges.src['n'], edges.dst['n'], edges.src['f'], edges.dst['f']],
                        dim=-1
                    )
                }
            )
            edge_input = graph.edata['cat']
        return self.model(edge_input)
    
class MeshEdgeBlockWithContext(nn.Module):
    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        input_dim_flow: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False
    ):
        super().__init__()
        self.edge_mlp = MeshGraphEdgeMLPConcatWithContext(
            efeat_dim=input_dim_edges,
            src_dim=input_dim_nodes,
            dst_dim=input_dim_nodes,
            flow_dim=input_dim_flow,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation
        )

    def forward(
        self,
        efeat: torch.Tensor,
        nfeat: torch.Tensor,
        flow_features: torch.Tensor,
        graph: DGLGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        efeat_new = self.edge_mlp(efeat, nfeat, flow_features, graph)
        efeat_new = efeat_new + efeat  # Residual connection
        return efeat_new, nfeat
    
class MeshNodeBlockWithContext(nn.Module):
    def __init__(
        self,
        aggregation: str,
        input_dim_nodes: int,
        input_dim_edges: int,
        input_dim_flow: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False
    ):
        super().__init__()
        self.aggregation = aggregation
        # Input to MLP includes aggregated edge features, node features, and flow features
        input_dim = input_dim_nodes + input_dim_edges + input_dim_flow
        self.node_mlp = MeshGraphMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation_fn=activation_fn,
            norm_type=norm_type,
            recompute_activation=recompute_activation
        )

    def forward(
        self,
        efeat: torch.Tensor,
        nfeat: torch.Tensor,
        flow_features: torch.Tensor,
        graph: DGLGraph
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cat_feat = aggregate_and_concat(efeat, nfeat, graph, self.aggregation)
        cat_feat = torch.cat([cat_feat, flow_features], dim=-1)
        nfeat_new = self.node_mlp(cat_feat) + nfeat  # Residual connection
        return efeat, nfeat_new
    
class MeshGraphNetProcessorWithContext(nn.Module):
    def __init__(
        self,
        processor_size: int,
        input_dim_node: int,
        input_dim_edge: int,
        input_dim_flow: int,
        num_layers_node: int,
        num_layers_edge: int,
        hidden_dim: int,
        aggregation: str = "sum",
        norm_type: str = "LayerNorm",
        activation_fn: nn.Module = nn.SiLU(),
        num_processor_checkpoint_segments: int = 0
    ):
        super().__init__()
        self.processor_size = processor_size
        # Create edge and node blocks
        edge_blocks = [
            MeshEdgeBlockWithContext(
                input_dim_nodes=input_dim_node,
                input_dim_edges=input_dim_edge,
                input_dim_flow=input_dim_flow,
                output_dim=input_dim_edge,
                hidden_dim=hidden_dim,
                hidden_layers=num_layers_edge,
                activation_fn=activation_fn,
                norm_type=norm_type
            ) for _ in range(processor_size)
        ]
        node_blocks = [
            MeshNodeBlockWithContext(
                aggregation=aggregation,
                input_dim_nodes=input_dim_node,
                input_dim_edges=input_dim_edge,
                input_dim_flow=input_dim_flow,
                output_dim=input_dim_node,
                hidden_dim=hidden_dim,
                hidden_layers=num_layers_node,
                activation_fn=activation_fn,
                norm_type=norm_type
            ) for _ in range(processor_size)
        ]
        self.layers = nn.ModuleList([b for pair in zip(edge_blocks, node_blocks) for b in pair])
        # Checkpointing (simplified; add if needed)
        self.checkpoint_segments = (
            [(0, processor_size * 2)] if num_processor_checkpoint_segments == 0
            else torch.chunk(torch.arange(processor_size * 2), num_processor_checkpoint_segments)
        )

    def run_function(self, start: int, end: int):
        def custom_forward(
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            flow_features: torch.Tensor,
            graph: DGLGraph
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            efeat, nfeat = edge_features, node_features
            for i in range(start, end):
                efeat, nfeat = self.layers[i](efeat, nfeat, flow_features, graph)
            return efeat, nfeat, flow_features
        return custom_forward

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        flow_features: torch.Tensor,
        graph: DGLGraph
    ) -> torch.Tensor:
        efeat, nfeat, ffeat = edge_features, node_features, flow_features
        for start, end in self.checkpoint_segments:
            efeat, nfeat, ffeat = self.run_function(start, end)(nfeat, efeat, ffeat, graph)
        return nfeat
    
class MeshGraphNetWithContext(nn.Module):
    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        input_dim_context: int,
        output_dim: int,
        processor_size: int = 15,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: int = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: int = 2,
        hidden_dim_context_encoder: int = 128,
        num_layers_context_encoder: int = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: int = 2,
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        aggregation: str = "sum",
        mlp_activation_fn: str = "silu",
        norm_type: str = "LayerNorm",
        do_concat_trick=False,
        num_processor_checkpoint_segments: int = 0
    ):
        super().__init__()
        activation_fn = get_activation(mlp_activation_fn)
        
        # Encoders
        self.node_encoder = MeshGraphMLP(
            input_dim=input_dim_nodes,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_node_encoder,
            hidden_layers=num_layers_node_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type
        )
        self.edge_encoder = MeshGraphMLP(
            input_dim=input_dim_edges,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_edge_encoder,
            hidden_layers=num_layers_edge_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type
        )
        self.context_encoder = MeshGraphMLP(
            input_dim=input_dim_context,
            output_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_context_encoder,
            hidden_layers=num_layers_context_encoder,
            activation_fn=activation_fn,
            norm_type=norm_type
        )
        
        # Processor
        self.processor = MeshGraphNetProcessorWithContext(
            processor_size=processor_size,
            input_dim_node=hidden_dim_processor,
            input_dim_edge=hidden_dim_processor,
            input_dim_flow=hidden_dim_processor,
            num_layers_node=num_layers_node_processor,
            num_layers_edge=num_layers_edge_processor,
            hidden_dim=hidden_dim_processor,
            aggregation=aggregation,
            norm_type=norm_type,
            activation_fn=activation_fn,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments
        )
        
        # Decoder
        self.node_decoder = MeshGraphMLP(
            input_dim=hidden_dim_processor,
            output_dim=output_dim,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=norm_type
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        context_features: torch.Tensor,
        graph: DGLGraph
    ) -> torch.Tensor:
        # Encode features
        node_features = self.node_encoder(node_features)
        edge_features = self.edge_encoder(edge_features)
        flow_features = self.context_encoder(context_features)
        
        # Process through GNN layers
        node_features = self.processor(node_features, edge_features, flow_features, graph)
        
        # Decode to output
        output = self.node_decoder(node_features)
        return output
    
g = dgl.graph(([0, 1], [1, 2]))  # 3 nodes, 2 edges
nfeat = torch.randn(3, 10)
efeat = torch.randn(2, 5)
cfeat = torch.randn(3, 8)
model = MeshGraphNetWithContext(input_dim_nodes=10, input_dim_edges=5, input_dim_context=8, output_dim=2)
out = model(nfeat, efeat, cfeat, g)
print(out.shape)