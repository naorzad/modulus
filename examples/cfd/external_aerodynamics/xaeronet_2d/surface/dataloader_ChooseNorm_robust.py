# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom dataset and dataloader for partitioned graph data in .bin files.
Loads NACA0012 graph partitions, normalizes node and edge features using
mean/std or median/IQR from a stats dictionary, with optional log transformation
for pressure and shear stress. Supports parallel loading and DDP.
"""

import os
import sys
import torch
import dgl
import json
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
import numpy as np

# Add parent directory to path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_bin_files


class GraphDataset(Dataset):
    """
    Dataset for loading and normalizing graph partitions from .bin files.

    Args:
        file_list (list): Paths to .bin files containing graph partitions.
        stats (dict): Dictionary with 'mean', 'std', 'median', 'iqr' for each feature.
        normalize (dict): Config for normalization per field (apply, method, log_transform).
    """
    def __init__(self, file_list, stats, normalize):
        self.file_list = file_list
        self.stats = stats
        self.normalize = normalize

        # Pre-convert stats to tensors for efficiency
        self.stat_tensors = {}
        for stat_type in ["mean", "std", "median", "iqr"]:
            self.stat_tensors[stat_type] = {
                field: torch.tensor(values) for field, values in stats[stat_type].items()
            }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        run_id = os.path.basename(file_path).split("_")[-1].split(".")[0]  # Extract run ID

        try:
            graphs, _ = dgl.load_graphs(file_path)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, run_id

        normalized_partitions = []
        for graph in graphs:
            # Normalize node and edge data
            for field, config in self.normalize.items():
                if not config.get("apply", False) or field not in (graph.ndata if field != "x" else graph.edata):
                    continue
                
                data = graph.ndata[field] if field != "x" else graph.edata[field]
                method = config.get("method", "mean_std")
                log_transform = config.get("log_transform", False)

                if log_transform and field in ["pressure", "shear_stress"]:
                    data = torch.sign(data) * torch.log1p(torch.abs(data))

                if method == "mean_std":
                    data = (data - self.stat_tensors["mean"][field]) / self.stat_tensors["std"][field]
                else:  # median_iqr
                    data = (data - self.stat_tensors["median"][field]) / self.stat_tensors["iqr"][field]
                
                if field != "x":
                    graph.ndata[field] = data
                else:
                    graph.edata[field] = data

            normalized_partitions.append(graph)

        return normalized_partitions, run_id


def create_dataloader(file_list, stats, normalize, batch_size=1, shuffle=False, use_ddp=True,
                      drop_last=True, num_workers=4, pin_memory=True, prefetch_factor=None):
    """
    Create a GraphDataLoader for partitioned graph data with flexible normalization.

    Args:
        file_list (list): Paths to .bin files.
        stats (dict): Stats dictionary with 'mean', 'std', 'median', 'iqr'.
        normalize (dict): Normalization settings per field (apply, method, log_transform).
        batch_size (int): Samples per batch.
        shuffle (bool): Shuffle data.
        use_ddp (bool): Use DistributedDataParallel.
        drop_last (bool): Drop last incomplete batch.
        num_workers (int): Worker processes for loading.
        pin_memory (bool): Use CUDA pinned memory.
        prefetch_factor (int, optional): Prefetch batches per worker.

    Returns:
        GraphDataLoader: Configured dataloader.
    """
    dataset = GraphDataset(file_list, stats, normalize)
    return GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        use_ddp=use_ddp
    )


if __name__ == "__main__":
    data_path = "partitions"
    stats_file = "global_stats.json"

    with open(stats_file, "r") as f:
        stats = json.load(f)

    file_list = find_bin_files(data_path)
    normalize = {
        "coordinates": {"apply": True, "method": "mean_std", "log_transform": False},
        "normals": {"apply": True, "method": "mean_std", "log_transform": False},
        "area": {"apply": True, "method": "mean_std", "log_transform": False},
        "pressure": {"apply": True, "method": "median_iqr", "log_transform": True},
        "shear_stress": {"apply": True, "method": "median_iqr", "log_transform": True},
        "x": {"apply": True, "method": "mean_std", "log_transform": False}
    }

    dataloader = create_dataloader(
        file_list, stats, normalize, batch_size=1, use_ddp=False, num_workers=1
    )

    for batch_partitions, label in dataloader:
        for graph in batch_partitions:
            print(f"Graph: {graph}")
        print(f"Label: {label}")
        break  # Test one batch