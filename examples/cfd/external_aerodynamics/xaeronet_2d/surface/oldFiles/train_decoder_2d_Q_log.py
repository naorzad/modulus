# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed training pipeline for MeshGraphNet on partitioned NACA0012 graph data.
Loads .bin files, normalizes features with flexible mean/std or median/IQR stats,
trains in parallel with DistributedDataParallel, and validates with point cloud outputs.
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
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from modulus.models.layers import get_activation
from dataloader_ChooseNorm_robust import create_dataloader
from utils import find_bin_files, save_checkpoint, load_checkpoint, count_trainable_params

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

class MeshGraphNetWithContext(MeshGraphNet):
    def __init__(self, input_dim_nodes, input_dim_edges, output_dim, context_dim=4, processor_size=15,
                 mlp_activation_fn="relu", num_layers_node_processor=2, num_layers_edge_processor=2,
                 hidden_dim_processor=64, hidden_dim_node_encoder=64, num_layers_node_encoder=2,
                 hidden_dim_edge_encoder=64, num_layers_edge_encoder=2, hidden_dim_node_decoder=128,
                 num_layers_node_decoder=2, aggregation="sum", do_concat_trick=False,
                 num_processor_checkpoint_segments=0, recompute_activation=False, dropout_rate=0.3):
        super().__init__(
            input_dim_nodes=input_dim_nodes, input_dim_edges=input_dim_edges, output_dim=output_dim,
            processor_size=processor_size, mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor, num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor, hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder, hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder, hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder, aggregation=aggregation,
            do_concat_trick=do_concat_trick, num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            recompute_activation=recompute_activation
        )
        self.context_dim = context_dim
        self.dropout_rate = dropout_rate
        act_fn = get_activation(mlp_activation_fn)
        self.node_decoder = MeshGraphMLP(
            hidden_dim_processor + context_dim, output_dim=output_dim, hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder, activation_fn=act_fn, norm_type=None,
            recompute_activation=recompute_activation
        )

    def forward(self, node_features, edge_features, graph, context):
        if edge_features.shape[1] != self.edge_encoder.model[0].in_features:
            raise ValueError(f"Expected edge_features with {self.edge_encoder.model[0].in_features} features, got {edge_features.shape[1]}")
        
        dtype = self.node_encoder.model[0].weight.dtype
        node_features, edge_features, context = (node_features.to(dtype), edge_features.to(dtype), context.to(dtype))
        
        edge_features = F.dropout(self.edge_encoder(edge_features), p=self.dropout_rate, training=self.training)
        node_features = F.dropout(self.node_encoder(node_features), p=self.dropout_rate, training=self.training)
        node_features = self.processor(node_features, edge_features, graph)
        
        context_expanded = context.unsqueeze(0).expand(node_features.size(0), -1)
        node_features = torch.cat([node_features, context_expanded], dim=1)
        return F.dropout(self.node_decoder(node_features), p=self.dropout_rate, training=self.training)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = cfg.enable_cudnn_benchmark
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    print(f"Rank {dist.rank} of {dist.world_size}")

    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
        initialize_wandb(project="naca0012", entity="Modulus", name="naca0012", mode="disabled", group="group", save_code=True)

    amp_dtype, amp_device = torch.bfloat16, "cuda"
    train_dataset = find_bin_files(to_absolute_path(cfg.partitions_path))
    valid_dataset = find_bin_files(to_absolute_path(cfg.validation_partitions_path))

    # Load all stats
    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)

    # Define normalization settings before dataloader calls
    normalize = {
        "coordinates": {"apply": True, "method": "mean_std", "log_transform": False},
        "normals": {"apply": True, "method": "mean_std", "log_transform": False},
        "area": {"apply": True, "method": "mean_std", "log_transform": False},
        "pressure": {"apply": True, "method": "mean_std", "log_transform": False},
        "shear_stress": {"apply": True, "method": "mean_std", "log_transform": False}, #median_iqr
        "x": {"apply": True, "method": "mean_std", "log_transform": False}
    }

    # Create dataloaders with normalize
    train_dataloader = create_dataloader(
        train_dataset, stats, normalize, batch_size=100, prefetch_factor=None, use_ddp=True, num_workers=4
    )
    graphs = [graph_partitions for graph_partitions, _ in train_dataloader]

    if dist.rank == 0:
        validation_dataloader = create_dataloader(
            valid_dataset, stats, normalize, batch_size=1, prefetch_factor=None, use_ddp=False, num_workers=4
        )
        validation_graphs = [graph_partitions for graph_partitions, _ in validation_dataloader]
        validation_ids = [id[0] for _, id in validation_dataloader]
        print(f"Training dataset size: {len(graphs)*dist.world_size}, Validation size: {len(validation_dataloader)}")

    # Model setup
    model = MeshGraphNetWithContext(
        input_dim_nodes=16, input_dim_edges=3, output_dim=3, context_dim=4,
        processor_size=cfg.num_message_passing_layers, aggregation="sum",
        hidden_dim_node_encoder=cfg.hidden_dim, hidden_dim_edge_encoder=cfg.hidden_dim,
        hidden_dim_node_decoder=cfg.hidden_dim, mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick, num_processor_checkpoint_segments=cfg.checkpoint_segments,
        dropout_rate=0.3
    ).to(device)

    if dist.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[dist.local_rank], output_device=dist.device,
                                        broadcast_buffers=dist.broadcast_buffers, find_unused_parameters=dist.find_unused_parameters,
                                        gradient_as_bucket_view=True, static_graph=True)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=20, eta_min=1e-8)
    scaler = GradScaler('cuda')

    start_epoch, _ = load_checkpoint(model, optimizer, scaler, scheduler, cfg.checkpoint_filename)

    # Training loop
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        total_loss = 0
        for i in range(len(graphs)):
            optimizer.zero_grad()
            subgraphs = graphs[i]
            for j in range(cfg.num_partitions):
                with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                    part = subgraphs[j].to(device)
                    ndata = torch.cat((
                        part.ndata["coordinates"], part.ndata["normals"],
                        torch.sin(2 * np.pi * part.ndata["coordinates"]), torch.cos(2 * np.pi * part.ndata["coordinates"]),
                        torch.sin(4 * np.pi * part.ndata["coordinates"]), torch.cos(4 * np.pi * part.ndata["coordinates"]),
                        torch.sin(8 * np.pi * part.ndata["coordinates"]), torch.cos(8 * np.pi * part.ndata["coordinates"]),
                    ), dim=1)
                    context_file = os.path.join(cfg.partitions_path, f"global_context_{i+1}.npy")
                    global_context = torch.tensor(np.load(context_file), device=device)
                    pred = model(ndata, part.edata["x"], part, global_context)
                    pred_filtered = pred[part.ndata["inner_node"].bool(),:]
                    target = torch.cat((part.ndata["pressure"], part.ndata["shear_stress"]), dim=1)
                    target_filtered = target[part.ndata["inner_node"].bool()]
                    loss = torch.mean((pred_filtered - target_filtered) ** 2) / cfg.num_partitions
                    total_loss += loss.item()
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 32.0)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if dist.rank == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}, LR: {lr}, Total Loss: {total_loss / len(graphs)}")
            writer.add_scalar("training_loss", total_loss / len(graphs), epoch)
            writer.add_scalar("learning_rate", lr, epoch)

            if epoch % cfg.save_checkpoint_freq == 0:
                save_checkpoint(model, optimizer, scaler, scheduler, epoch + 1, loss.item(), cfg.checkpoint_filename)

        # Validation
        if dist.rank == 0 and epoch % cfg.validation_freq == 0:
            model.eval()
            valid_loss = 0
            for i in range(len(validation_graphs)):
                all_inner_node_ids = set()
                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j]
                    all_inner_node_ids.update(part.ndata[dgl.NID][part.ndata["inner_node"].bool()].cpu().numpy())
                
                num_inner_nodes = len(all_inner_node_ids)
                nid_to_idx = {nid: idx for idx, nid in enumerate(sorted(all_inner_node_ids))}
                
#                Use float32 consistently for accumulators
                preds = {k: torch.zeros((num_inner_nodes, v), dtype=torch.float32, device=device) 
                         for k, v in [("pressure", 1), ("shear_stress", 2), ("pressure_true", 1), ("shear_stress_true", 2),
                                      ("coordinates", 2), ("normals", 2), ("area", 1)]}

                for j in range(cfg.num_partitions):
                    part = validation_graphs[i][j].to(device)
                    ndata = torch.cat((
                        part.ndata["coordinates"], part.ndata["normals"],
                        torch.sin(2 * np.pi * part.ndata["coordinates"]), torch.cos(2 * np.pi * part.ndata["coordinates"]),
                        torch.sin(4 * np.pi * part.ndata["coordinates"]), torch.cos(4 * np.pi * part.ndata["coordinates"]),
                        torch.sin(8 * np.pi * part.ndata["coordinates"]), torch.cos(8 * np.pi * part.ndata["coordinates"]),
                    ), dim=1)
                    context_file = os.path.join(cfg.validation_partitions_path, f"global_context_{i+1}.npy")
                    global_context = torch.tensor(np.load(context_file), device=device)
                    
                    with torch.no_grad():
                        with torch.autocast(amp_device, enabled=True, dtype=amp_dtype):
                            pred = model(ndata, part.edata["x"], part, global_context)
                            pred_filt = pred[part.ndata["inner_node"].bool(),:]
                            target = torch.cat((part.ndata["pressure"], part.ndata["shear_stress"]), dim=1)
                            target_filt = target[part.ndata["inner_node"].bool()]
                            loss = torch.mean((pred_filt - target_filt) ** 2) / cfg.num_partitions
                            valid_loss += loss.item()

                            inner_nodes = part.ndata[dgl.NID][part.ndata["inner_node"].bool()]
                            indices = torch.tensor([nid_to_idx[nid.item()] for nid in inner_nodes], device=device)
                            preds["pressure"][indices] = pred_filt[:, 0:1].to(torch.float32)
                            preds["shear_stress"][indices] = pred_filt[:, 1:].to(torch.float32)
                            preds["pressure_true"][indices] = target_filt[:, 0:1].to(torch.float32)
                            preds["shear_stress_true"][indices] = target_filt[:, 1:].to(torch.float32)
                            preds["coordinates"][indices] = part.ndata["coordinates"][part.ndata["inner_node"].bool()].to(torch.float32)
                            preds["normals"][indices] = part.ndata["normals"][part.ndata["inner_node"].bool()].to(torch.float32)
                            preds["area"][indices] = part.ndata["area"][part.ndata["inner_node"].bool()].to(torch.float32)

                # Denormalize with flexibility
                for field in preds:
                    if normalize.get(field, {}).get("apply", False):
                        method = normalize[field]["method"]
                        log_transform = normalize.get(field, {}).get("log_transform", False)
                        if method == "mean_std":
                            denorm = preds[field].cpu() * torch.tensor(stats["std"][field]) + torch.tensor(stats["mean"][field])
                        else:  # median_iqr
                            denorm = preds[field].cpu() * torch.tensor(stats["iqr"][field]) + torch.tensor(stats["median"][field])
                        if log_transform and field in ["pressure", "shear_stress", "pressure_true", "shear_stress_true"]:
                            denorm = torch.sign(denorm) * (torch.exp(torch.abs(denorm)) - 1)
                        preds[field] = denorm
                        # Debug: Print denorm ranges
                        print(f"{field} denorm min: {preds[field].min().item()}, max: {preds[field].max().item()}")
                    else:
                        preds[field] = preds[field].cpu()
                        if field in ["pressure", "shear_stress", "pressure_true", "shear_stress_true"] and normalize.get(field, {}).get("log_transform", False):
                            preds[field] = torch.sign(preds[field]) * (torch.exp(torch.abs(preds[field])) - 1)
                        print(f"{field} no-norm min: {preds[field].min().item()}, max: {preds[field].max().item()}")

                # Line ~206-207: Add debug for training target normalization
                target_filt = target[part.ndata["inner_node"].bool()].to(torch.float32)
                print(f"Epoch {epoch+1}, Graph {i}, Part {j} - Target shear_stress min: {target_filt[:, 1:].min().item()}, max: {target_filt[:, 1:].max().item()}")
                
                # Save point cloud
                coords_3d = np.column_stack((preds["coordinates"].numpy(), np.zeros(num_inner_nodes)))
                pc = pv.PolyData(coords_3d)
                for field in preds:
                    pc[field] = preds[field].numpy() if field.startswith("shear") else preds[field].numpy().squeeze(-1) if preds[field].shape[-1] == 1 else preds[field].numpy()
                pc.save(f"point_cloud_{validation_ids[i]}.vtp")

            print(f"Epoch {epoch+1}, Validation Error: {valid_loss / len(validation_graphs)}")
            writer.add_scalar("validation_loss", valid_loss / len(validation_graphs), epoch)

    if dist.rank == 0:
        save_checkpoint(model, optimizer, scaler, scheduler, cfg.num_epochs, loss.item(), "final_model_checkpoint.pth")
        print("Training complete")

if __name__ == "__main__":
    main()