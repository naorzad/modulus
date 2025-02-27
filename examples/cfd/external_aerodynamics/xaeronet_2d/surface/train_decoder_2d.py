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
which operates on partitioned graph data for the AWS drivaer dataset. It includes
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
        processor_size=15,
        mlp_activation_fn="relu",
        num_layers_node_processor=2,
        num_layers_edge_processor=2,
        hidden_dim_processor=64,
        hidden_dim_node_encoder=64,
        num_layers_node_encoder=2,
        hidden_dim_edge_encoder=64,
        num_layers_edge_encoder=2,
        hidden_dim_node_decoder=128,
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
        
        # Expand context and concatenate with node features before decoding
        context_expanded = context.unsqueeze(0).expand(node_features.size(0), -1)
        node_features = torch.cat([node_features, context_expanded], dim=1)
        
        # Decode node features with dropout
        node_features = F.dropout(self.node_decoder(node_features), p=self.dropout_rate, training=self.training)
        return node_features
    
from dataloader import create_dataloader
from utils import (
    find_bin_files,
    save_checkpoint,
    load_checkpoint,
    count_trainable_params,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    # Enable cuDNN auto-tuner
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    # Instantiate the distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    # Instantiate the writers
    if dist.rank == 0:
        print("Initializing TensorBoard and WandB logging...")
        writer = SummaryWriter(log_dir="tensorboard")
        initialize_wandb(
            project="naca0012",
            entity="Modulus",
            name="naca0012",
            mode="disabled",
            group="group",
            save_code=True,
        )
        print("Logging initialized.")

    # AMP Configs
    amp_dtype = torch.bfloat16
    amp_device = "cuda"

    # Find all .bin files in the directory
    train_dataset = find_bin_files(to_absolute_path(cfg.partitions_path))
    valid_dataset = find_bin_files(to_absolute_path(cfg.validation_partitions_path))

    # Prepare the stats
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std_dev"]

    # Define normalization settings
    normalize = {
        "coordinates": True,
        "normals": True,
        "area": True,
        "pressure": True,
        "shear_stress": True,
        "x": True
}

    # Create DataLoader
    print("Before DataLoader creation")
    # Create DataLoader with normalization settings
    train_dataloader = create_dataloader(
        train_dataset,
        mean,
        std,
        batch_size=1,
        prefetch_factor=None,
        use_ddp=True,
        num_workers=4,
    )
    print("After DataLoader creation")
    # graphs is a list of graphs, each graph is a list of partitions
    graphs = [graph_partitions for graph_partitions, _ in train_dataloader]

    if dist.rank == 0:
        validation_dataloader = create_dataloader(
            valid_dataset,
            mean,
            std,
            batch_size=1,
            prefetch_factor=None,
            use_ddp=False,
            num_workers=4,
        )
        validation_graphs = [
            graph_partitions for graph_partitions, _ in validation_dataloader
        ]
        validation_ids = [id[0] for _, id in validation_dataloader]
        print(f"Training dataset size: {len(graphs)*dist.world_size}")
        print(f"Validation dataset size: {len(validation_dataloader)}")

    ######################################
    # Training #
    ######################################

    # Initialize model
    model = MeshGraphNetWithContext(  # Updated class
        input_dim_nodes=16,
        input_dim_edges=3,
        output_dim=3,
        context_dim=3,  # Add context dimension
        processor_size=cfg.num_message_passing_layers,
        aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick,
        num_processor_checkpoint_segments=cfg.checkpoint_segments,
        dropout_rate = 0.3
    ).to(device)
    print(f"Number of trainable parameters: {count_trainable_params(model)}")

    # DistributedDataParallel wrapper
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-8)
    
    scaler = GradScaler('cuda')
    print("Instantiated the model and optimizer")

    # Check if there's a checkpoint to resume from
    start_epoch, _ = load_checkpoint(
        model, optimizer, scaler, scheduler, cfg.checkpoint_filename
    )

    # Training loop
    print("Training started")
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0
        for i in range(len(graphs)):
            optimizer.zero_grad()
            subgraphs = graphs[i]  # Get the partitions of the graph
            for j in range(cfg.num_partitions):
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = subgraphs[j].to(device)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(4 * np.pi * part.ndata["coordinates"]),
                            torch.cos(4 * np.pi * part.ndata["coordinates"]),
                            torch.sin(8 * np.pi * part.ndata["coordinates"]),
                            torch.cos(8 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )
                    # Load global context for the current partition
                    context_file_path = os.path.join(cfg.partitions_path, f"global_context_{i+1}.npy")
                    global_context = torch.tensor(np.load(context_file_path), device=device)
                    
                    # Add global context embedding
                    pred = model(ndata, part.edata["x"], part, global_context)
                    pred_filtered = pred[part.ndata["inner_node"].bool(), :]
                    target = torch.cat(
                        (part.ndata["pressure"], part.ndata["shear_stress"][:, :2]), dim=1  # Ensure shear_stress is 2D
                    )
                    target_filtered = target[part.ndata["inner_node"].bool()]
                    loss = (
                        torch.mean((pred_filtered - target_filtered) ** 2)
                        / cfg.num_partitions
                    )
                    total_loss += loss.item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Log the training loss
        if dist.rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}, Learning Rate: {current_lr}, Total Loss: {total_loss / len(graphs)}"
            )
            writer.add_scalar("training_loss", total_loss / len(graphs), epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

        # Save checkpoint periodically
        if (epoch) % cfg.save_checkpoint_freq == 0:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    scheduler,
                    epoch + 1,
                    loss.item(),
                    cfg.checkpoint_filename,
                )

        ######################################
        # Validation #
        ######################################

        if dist.rank == 0 and epoch % cfg.validation_freq == 0:
            valid_loss = 0

            for i in range(len(validation_graphs)):
                # Placeholder to accumulate predictions and node features for the full graph's nodes
                num_nodes = sum(
                    [subgraph.num_nodes() for subgraph in validation_graphs[i]]
                )

                # Initialize accumulators for predictions and node features
                pressure_pred = torch.zeros(
                    (num_nodes, 1), dtype=torch.float32, device=device
                )
                shear_stress_pred = torch.zeros(
                    (num_nodes, 2), dtype=torch.float32, device=device  # Change to 2D
                )
                pressure_true = torch.zeros(
                    (num_nodes, 1), dtype=torch.float32, device=device
                )
                shear_stress_true = torch.zeros(
                    (num_nodes, 2), dtype=torch.float32, device=device  # Change to 2D
                )
                coordinates = torch.zeros(
                    (num_nodes, 2), dtype=torch.float32, device=device  # Change to 2D
                )
                normals = torch.zeros(
                    (num_nodes, 2), dtype=torch.float32, device=device  # Change to 2D
                )
                area = torch.zeros((num_nodes, 1), dtype=torch.float32, device=device)

                # Accumulate predictions and node features from all partitions
                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j].to(device)

                    # Get node features (coordinates and normals)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(4 * np.pi * part.ndata["coordinates"]),
                            torch.cos(4 * np.pi * part.ndata["coordinates"]),
                            torch.sin(8 * np.pi * part.ndata["coordinates"]),
                            torch.cos(8 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )

                    with torch.no_grad():
                        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                            # Load global context for the current partition
                            context_file_path = os.path.join(cfg.validation_partitions_path, f"global_context_{i+1}.npy")
                            global_context = torch.tensor(np.load(to_absolute_path(context_file_path)), device=device)
                            
                            # Add global context embedding
                            pred = model(ndata, part.edata["x"], part, global_context)
                            pred_filtered = pred[part.ndata["inner_node"].bool()]
                            target = torch.cat(
                                (part.ndata["pressure"], part.ndata["shear_stress"]),
                                dim=1,
                            )
                            target_filtered = target[part.ndata["inner_node"].bool()]
                            loss = (
                                torch.mean((pred_filtered - target_filtered) ** 2)
                                / cfg.num_partitions
                            )
                            valid_loss += loss.item()

                            # Store the predictions based on the original node IDs (using `dgl.NID`)
                            original_nodes = part.ndata[dgl.NID]
                            inner_original_nodes = original_nodes[
                                part.ndata["inner_node"].bool()
                            ]

                            # Accumulate the predictions
                            pressure_pred[inner_original_nodes] = (
                                pred_filtered[:, 0:1].clone().to(torch.float32)
                            )
                            shear_stress_pred[inner_original_nodes] = (
                                pred_filtered[:, 1:].clone().to(torch.float32)
                            )

                            # Accumulate the ground truth
                            pressure_true[inner_original_nodes] = (
                                target_filtered[:, 0:1].clone().to(torch.float32)
                            )
                            shear_stress_true[inner_original_nodes] = (
                                target_filtered[:, 1:].clone().to(torch.float32)
                            )

                            # Accumulate the node features
                            coordinates[original_nodes] = (
                                part.ndata["coordinates"].clone().to(torch.float32)
                            )
                            normals[original_nodes] = (
                                part.ndata["normals"].clone().to(torch.float32)
                            )
                            area[original_nodes] = (
                                part.ndata["area"].clone().to(torch.float32)
                            )

                # Denormalize predictions and node features using the global stats
                if normalize.get("pressure", True):
                    pressure_pred_denorm = (
                        pressure_pred.cpu() * torch.tensor(std["pressure"])
                    ) + torch.tensor(mean["pressure"])
                    pressure_true_denorm = (
                        pressure_true.cpu() * torch.tensor(std["pressure"])
                    ) + torch.tensor(mean["pressure"])
                else:
                    pressure_pred_denorm = pressure_pred.cpu()
                    pressure_true_denorm = pressure_true.cpu()

                if normalize.get("shear_stress", True):
                    shear_stress_pred_denorm = (
                        shear_stress_pred.cpu() * torch.tensor(std["shear_stress"])
                    ) + torch.tensor(mean["shear_stress"])
                    shear_stress_true_denorm = (
                        shear_stress_true.cpu() * torch.tensor(std["shear_stress"])
                    ) + torch.tensor(mean["shear_stress"])
                else:
                    shear_stress_pred_denorm = shear_stress_pred.cpu()
                    shear_stress_true_denorm = shear_stress_true.cpu()

                if normalize.get("coordinates", True):
                    coordinates_denorm = (
                        coordinates.cpu() * torch.tensor(std["coordinates"])
                    ) + torch.tensor(mean["coordinates"])
                else:
                    coordinates_denorm = coordinates.cpu()

                if normalize.get("normals", True):
                    normals_denorm = (
                        normals.cpu() * torch.tensor(std["normals"])
                    ) + torch.tensor(mean["normals"])
                else:
                    normals_denorm = normals.cpu()

                if normalize.get("area", True):
                    area_denorm = (area.cpu() * torch.tensor(std["area"])) + torch.tensor(mean["area"])
                else:
                    area_denorm = area.cpu()

                # Add zero Z-coordinate to coordinates, normals, and shear_stress
                coordinates_denorm = np.column_stack((coordinates_denorm.numpy(), np.zeros((coordinates_denorm.shape[0], 1))))
                normals_denorm = np.column_stack((normals_denorm.numpy(), np.zeros((normals_denorm.shape[0], 1))))
                shear_stress_pred_denorm = np.column_stack((shear_stress_pred_denorm.numpy(), np.zeros((shear_stress_pred_denorm.shape[0], 1))))
                shear_stress_true_denorm = np.column_stack((shear_stress_true_denorm.numpy(), np.zeros((shear_stress_true_denorm.shape[0], 1))))

                # Save the full point cloud after accumulating all partition predictions
                # Create a PyVista PolyData object for the point cloud
                point_cloud = pv.PolyData(coordinates_denorm)
                point_cloud["coordinates"] = coordinates_denorm
                point_cloud["normals"] = normals_denorm
                point_cloud["area"] = area_denorm.numpy()
                point_cloud["pressure_pred"] = pressure_pred_denorm.numpy()
                point_cloud["shear_stress_pred"] = shear_stress_pred_denorm
                point_cloud["pressure_true"] = pressure_true_denorm.numpy()
                point_cloud["shear_stress_true"] = shear_stress_true_denorm

                # Save the point cloud
                point_cloud.save(f"point_cloud_{validation_ids[i]}.vtp")

            print(
                f"Epoch {epoch+1}, Validation Error: {valid_loss / len(validation_graphs)}"
            )
            writer.add_scalar(
                "validation_loss", valid_loss / len(validation_graphs), epoch
            )

    # Save final checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        save_checkpoint(
            model,
            optimizer,
            scaler,
            scheduler,
            cfg.num_epochs,
            loss.item(),
            "final_model_checkpoint.pth",
        )
        print("Training complete")


if __name__ == "__main__":
    main()
