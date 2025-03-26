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
import json
import dgl
import pyvista as pv
import torch
import hydra
import numpy as np
from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from modulus.distributed import DistributedManager
from modulus.launch.logging import initialize_wandb
from modulus.models.meshgraphnet import MeshGraphNet 
from modulus.models.gnn_layers.mesh_edge_block import MeshEdgeBlock
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.gnn_layers.mesh_node_block import MeshNodeBlock
from torch import Tensor
from dgl import DGLGraph
from modulus.models.layers import get_activation

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
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor + context_dim,  # Adjust for context dimension
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
        context = context.to(self.node_decoder.model[0].weight.dtype)
        
        # Encode node and edge features with dropout
        edge_features = F.dropout(self.edge_encoder(edge_features), p=self.dropout_rate, training=self.training)
        node_features = F.dropout(self.node_encoder(node_features), p=self.dropout_rate, training=self.training)
        
        # Process node and edge features
        node_features = self.processor(node_features, edge_features, graph)
        
        # # Expand context and concatenate with node features before decoding
        context_expanded = context.unsqueeze(0).expand(node_features.size(0), -1)
        node_features = torch.cat([node_features, context_expanded], dim=1)
        
        # Decode node features with dropout
        node_features = F.dropout(self.node_decoder(node_features), p=self.dropout_rate, training=self.training)
        return node_features