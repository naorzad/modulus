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
This code defines a distributed training pipeline for training MeshGraphNetWithContext at scale,
which operates on partitioned graph data for the NACA 0012 dataset. It includes loading partitioned
graphs from .bin files, normalizing node and edge features using precomputed statistics, and
training the model in parallel using DistributedDataParallel across multiple GPUs. The training
loop incorporates a global context dimension, computes predictions for each graph partition,
calculates loss, and updates model parameters using mixed precision. Periodic checkpointing saves
the model and training progress, while validation compares predictions against ground truth,
saving results as point clouds. Metrics are logged to TensorBoard and optionally to Weights and Biases.
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
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from modulus.distributed import DistributedManager
from modulus.launch.logging import initialize_wandb
from meshgraphnetwithcontextGlobalEncoder import MeshGraphNetWithContext

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from dataloader_ChooseNorm_robust import create_dataloader
from utils import (
    find_bin_files,
    save_checkpoint,
    load_checkpoint,
    count_trainable_params,
)

# Loss function: Weighted MSE with spatial coordinate weighting
def loss_to_zero(pred, target, coordinates, inner_node_mask, num_partitions):
    pred_filtered = pred[inner_node_mask]
    target_filtered = target[inner_node_mask]
    
    x_coords_filtered = coordinates[:, 0][inner_node_mask]
    y_coords_filtered = coordinates[:, 1][inner_node_mask]
    
    pressure_loss = torch.sum((pred_filtered[:, 0] - target_filtered[:, 0]) ** 2, dim=0) * \
                    (1 / (x_coords_filtered + 1e-4)) * (1 / (y_coords_filtered + 1e-4))
    shear_loss = torch.sum((pred_filtered[:, 1:] - target_filtered[:, 1:]) ** 2, dim=0) * \
                 (1 / (x_coords_filtered.unsqueeze(1) + 1e-4)) * (1 / (y_coords_filtered.unsqueeze(1) + 1e-4))
    
    loss = 0.7 * torch.mean(pressure_loss) + 0.3 * torch.mean(shear_loss)
    weighted_loss = torch.sqrt(loss / num_partitions)
    
    return weighted_loss

# Alternative loss function: Weighted by target magnitude
def loss_func(pred, target, coordinates, inner_node_mask, num_partitions):
    pred_filtered = pred[inner_node_mask]
    target_filtered = target[inner_node_mask]
    
    pressure_loss = (pred_filtered[:, 0] - target_filtered[:, 0]) ** 2 * abs(target_filtered[:, 0])
    shear_loss = (pred_filtered[:, 1:] - target_filtered[:, 1:]) ** 2 * abs(target_filtered[:, 1:])
    
    loss = 0.7 * torch.mean(pressure_loss) + 0.3 * torch.mean(shear_loss)
    weighted_loss = loss / num_partitions
    
    return weighted_loss

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set number of partitions to 1 if not partitioning
    if not cfg.partition_graph:
        cfg.num_partitions = 1
    
    # Enable cuDNN auto-tuner for performance
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    # Set up logging for rank 0
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

    # Mixed precision settings
    amp_dtype = torch.bfloat16
    amp_device = "cuda"

    # Load datasets
    train_dataset = find_bin_files(to_absolute_path(cfg.partitions_path))
    valid_dataset = find_bin_files(to_absolute_path(cfg.validation_partitions_path))

    # Load precomputed statistics
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)

    # Normalization settings
    normalize = {
        "coordinates": {"apply": True, "method": "mean_std", "log_transform": False},
        "normals": {"apply": True, "method": "mean_std", "log_transform": False},
        "area": {"apply": True, "method": "mean_std", "log_transform": False},
        "pressure": {"apply": True, "method": "mean_std", "log_transform": False},
        "shear_stress": {"apply": True, "method": "mean_std", "log_transform": False},
        "x": {"apply": True, "method": "mean_std", "log_transform": False}
    }

    # Create data loaders
    print("Before DataLoader creation")
    train_dataloader = create_dataloader(
        train_dataset, stats, normalize, batch_size=1, prefetch_factor=None, use_ddp=True, num_workers=4
    )
    print("After DataLoader creation")
    graphs = [graph_partitions for graph_partitions, _ in train_dataloader]

    if dist.rank == 0:
        validation_dataloader = create_dataloader(
            valid_dataset, stats, normalize, batch_size=1, prefetch_factor=None, use_ddp=False, num_workers=4
        )
        validation_graphs = [graph_partitions for graph_partitions, _ in validation_dataloader]
        validation_ids = [id[0] for _, id in validation_dataloader]
        print(f"Training dataset size: {len(graphs)*dist.world_size}")
        print(f"Validation dataset size: {len(validation_dataloader)}")

    ### Training Setup ###

    # Initialize model with context dimension
    model = MeshGraphNetWithContext(
        input_dim_nodes=16,
        input_dim_edges=3,
        output_dim=3,
        input_dim_context=4,  # Define context dimension (adjust as needed)
        processor_size=cfg.num_message_passing_layers,
        aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim,
        hidden_dim_processor=cfg.hidden_dim*2,
        mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick,
        num_processor_checkpoint_segments=cfg.checkpoint_segments
    ).to(device)
    print(f"Number of trainable parameters: {count_trainable_params(model)}")

    # Wrap model with DistributedDataParallel if multi-GPU
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
    optimizer = optim.AdamW(model.parameters(), lr=cfg.start_lr, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3)
    scaler = GradScaler('cuda')
    print("Instantiated the model and optimizer")

    # Resume from checkpoint if available
    start_epoch, _ = load_checkpoint(
        model, optimizer, scaler, scheduler, cfg.checkpoint_filename
    )

    ### Training Loop ###
    print("Training started")
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0
        for i in range(len(graphs)):
            optimizer.zero_grad()
            subgraphs = graphs[i]
            for j in range(cfg.num_partitions):
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = subgraphs[j].to(device)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(1 * np.pi * part.ndata["coordinates"]),
                            torch.cos(1 * np.pi * part.ndata["coordinates"]),
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(3 * np.pi * part.ndata["coordinates"]),
                            torch.cos(3 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )
                    # Load global context for the current partition
                    context_file_path = os.path.join(cfg.partitions_path, f"global_context_{i+1}.npy")
                    global_context = torch.tensor(np.load(context_file_path), device=device)
                    
                    # Ensure global_context has shape (1, context_dim)
                    if global_context.dim() == 1:
                        global_context = global_context.unsqueeze(0)
                    
                    # Forward pass with context
                    pred = model(ndata, part.edata["x"], global_context, part)
                    pred_filtered = pred[part.ndata["inner_node"].bool()]
                    target = torch.cat((part.ndata["pressure"], part.ndata["shear_stress"]), dim=1)
                    target_filtered = target[part.ndata["inner_node"].bool()]
                    loss = torch.mean((pred_filtered - target_filtered) ** 2) / cfg.num_partitions
                    # loss = loss_to_zero(pred, target, part.ndata["coordinates"], 
                    #                   part.ndata["inner_node"].bool(), cfg.num_partitions)
                    total_loss += loss.item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step(total_loss)

        # Log training metrics
        if dist.rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}, Learning Rate: {current_lr}, Total Loss: {total_loss / len(graphs)}")
            writer.add_scalar("training_loss", total_loss / len(graphs), epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

        # Save checkpoint periodically
        if (epoch) % cfg.save_checkpoint_freq == 0:
            if dist.world_size > 1:
                torch.distributed.barrier()
            if dist.rank == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler, epoch + 1, 
                    loss.item(), cfg.checkpoint_filename
                )

        ### Validation Loop ###
        if dist.rank == 0 and epoch % cfg.validation_freq == 0:
            model.eval()
            valid_loss = 0
            for i in range(len(validation_graphs)):
                all_inner_node_ids = set()
                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j]
                    inner_mask = part.ndata["inner_node"].bool()
                    all_inner_node_ids.update(part.ndata[dgl.NID][inner_mask].cpu().numpy())
                
                num_inner_nodes = len(all_inner_node_ids)
                preds = {k: torch.zeros((num_inner_nodes, v), dtype=torch.float32, device=device) 
                         for k, v in [("pressure", 1), ("shear_stress", 2), ("pressure_true", 1), 
                                      ("shear_stress_true", 2), ("coordinates", 2), ("normals", 2), ("area", 1)]}
                nid_to_idx = {nid: idx for idx, nid in enumerate(sorted(all_inner_node_ids))}

                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j].to(device)
                    ndata = torch.cat(
                        (
                            part.ndata["coordinates"],
                            part.ndata["normals"],
                            torch.sin(1 * np.pi * part.ndata["coordinates"]),
                            torch.cos(1 * np.pi * part.ndata["coordinates"]),
                            torch.sin(2 * np.pi * part.ndata["coordinates"]),
                            torch.cos(2 * np.pi * part.ndata["coordinates"]),
                            torch.sin(16 * np.pi * part.ndata["coordinates"]),
                            torch.cos(16 * np.pi * part.ndata["coordinates"]),
                        ),
                        dim=1,
                    )
                    with torch.no_grad():
                        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                            context_file_path = os.path.join(cfg.validation_partitions_path, 
                                                           f"global_context_{i+1}.npy")
                            global_context = torch.tensor(np.load(context_file_path), device=device)
                            if global_context.dim() == 1:
                                global_context = global_context.unsqueeze(0)
                            
                            pred = model(ndata, part.edata["x"], global_context, part)
                            pred_filtered = pred[part.ndata["inner_node"].bool()]
                            target = torch.cat((part.ndata["pressure"], part.ndata["shear_stress"]), dim=1)
                            target_filtered = target[part.ndata["inner_node"].bool()]
                            loss = torch.mean((pred_filtered - target_filtered) ** 2) / cfg.num_partitions
                            # loss = loss_to_zero(pred, target, part.ndata["coordinates"], 
                            #                   part.ndata["inner_node"].bool(), cfg.num_partitions)
                            valid_loss += loss.item()

                            inner_nodes = part.ndata[dgl.NID][part.ndata["inner_node"].bool()]
                            indices = torch.tensor([nid_to_idx[nid.item()] for nid in inner_nodes], device=device)
                            preds["pressure"][indices] = pred_filtered[:, 0:1].to(torch.float32)
                            preds["shear_stress"][indices] = pred_filtered[:, 1:].to(torch.float32)
                            preds["pressure_true"][indices] = target_filtered[:, 0:1].to(torch.float32)
                            preds["shear_stress_true"][indices] = target_filtered[:, 1:].to(torch.float32)
                            preds["coordinates"][indices] = part.ndata["coordinates"][part.ndata["inner_node"].bool()].to(torch.float32)
                            preds["normals"][indices] = part.ndata["normals"][part.ndata["inner_node"].bool()].to(torch.float32)
                            preds["area"][indices] = part.ndata["area"][part.ndata["inner_node"].bool()].to(torch.float32)

                # Denormalize predictions
                for field in preds:
                    base_field = field.replace("_true", "") if "_true" in field else field
                    if normalize.get(base_field, {}).get("apply", False):
                        method = normalize[base_field]["method"]
                        log_transform = normalize[base_field]["log_transform"]
                        if method == "mean_std":
                            denorm = preds[field].cpu() * torch.tensor(stats["std"][base_field]) + \
                                    torch.tensor(stats["mean"][base_field])
                        else:  # median_iqr
                            denorm = preds[field].cpu() * torch.tensor(stats["iqr"][base_field]) + \
                                    torch.tensor(stats["median"][base_field])
                        if log_transform and field in ["pressure", "shear_stress", "pressure_true", "shear_stress_true"]:
                            denorm = torch.sign(denorm) * (torch.exp(torch.abs(denorm)) - 1)
                        preds[field] = denorm
                    else:
                        preds[field] = preds[field].cpu()

                # Save point cloud
                coords_3d = np.column_stack((preds["coordinates"].numpy(), np.zeros(num_inner_nodes)))
                pc = pv.PolyData(coords_3d)
                for field in preds:
                    pc[field] = preds[field].numpy() if field.startswith("shear") else \
                               preds[field].numpy().squeeze(-1) if preds[field].shape[-1] == 1 else preds[field].numpy()
                pc.save(f"point_cloud_{validation_ids[i]}.vtp")

            print(f"Epoch {epoch+1}, Validation Error: {valid_loss / len(validation_graphs)}")
            writer.add_scalar("validation_loss", valid_loss / len(validation_graphs), epoch)

    # Save final checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    if dist.rank == 0:
        save_checkpoint(
            model, optimizer, scaler, scheduler, cfg.num_epochs, 
            loss.item(), "final_model_checkpoint.pth"
        )
        print("Training complete")

if __name__ == "__main__":
    main()