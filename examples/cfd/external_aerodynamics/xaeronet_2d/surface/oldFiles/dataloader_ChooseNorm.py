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
This code defines a custom dataset class GraphDataset for loading and normalizing
graph partition data stored in .bin files. The dataset is initialized with a list
of file paths and global mean and standard deviation for node and edge attributes.
It normalizes node data (like coordinates, normals, pressure) and edge data based
on these statistics before returning the processed graph partitions and a corresponding
label (extracted from the file name). The code also provides a function create_dataloader
to create a data loader for efficient batch loading with configurable parameters such as
batch size, shuffle, and prefetching options. 
"""

import json
import torch
from torch.utils.data import Dataset
import os
import sys
import dgl
from dgl.dataloading import GraphDataLoader

# Get the absolute path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import find_bin_files


class GraphDataset(Dataset):
    """
    Custom dataset class for loading

    Parameters:
    ----------
        file_list (list of str): List of paths to .bin files containing partitions.
        min_vals (np.ndarray): Global minimum values for normalization.
        max_vals (np.ndarray): Global maximum values for normalization.
        normalize (dict): Dictionary specifying which data to normalize.
    """

    def __init__(self, file_list, min_vals, max_vals, normalize):
        self.file_list = file_list
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.normalize = normalize

        # Store normalization stats as tensors
        self.coordinates_min = torch.tensor(min_vals["coordinates"])
        self.coordinates_max = torch.tensor(max_vals["coordinates"])
        self.normals_min = torch.tensor(min_vals["normals"])
        self.normals_max = torch.tensor(max_vals["normals"])
        self.area_min = torch.tensor(min_vals["area"])
        self.area_max = torch.tensor(max_vals["area"])
        self.pressure_min = torch.tensor(min_vals["pressure"])
        self.pressure_max = torch.tensor(max_vals["pressure"])
        self.shear_stress_min = torch.tensor(min_vals["shear_stress"])
        self.shear_stress_max = torch.tensor(max_vals["shear_stress"])
        self.edge_x_min = torch.tensor(min_vals["x"])
        self.edge_x_max = torch.tensor(max_vals["x"])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        # Extract the ID from the file name
        file_name = os.path.basename(file_path)
        # Assuming file format is "graph_partitions_<run_id>.bin"
        run_id = file_name.split("_")[-1].split(".")[0]  # Extract the run ID

        # Load the partitioned graphs from the .bin file
        graphs, _ = dgl.load_graphs(file_path)

        # Process each partition (graph)
        normalized_partitions = []
        for graph in graphs:
            # Normalize node data using Min-Max scaling
            if self.normalize.get("coordinates", True):
                graph.ndata["coordinates"] = (
                    graph.ndata["coordinates"] - self.coordinates_min
                ) / (self.coordinates_max - self.coordinates_min)
            if self.normalize.get("normals", True):
                graph.ndata["normals"] = (
                    graph.ndata["normals"] - self.normals_min
                ) / (self.normals_max - self.normals_min)
            if self.normalize.get("area", True):
                graph.ndata["area"] = (
                    graph.ndata["area"] - self.area_min
                ) / (self.area_max - self.area_min)
            if self.normalize.get("pressure", True):
                graph.ndata["pressure"] = (
                    graph.ndata["pressure"] - self.pressure_min
                ) / (self.pressure_max - self.pressure_min)
            if self.normalize.get("shear_stress", True):
                graph.ndata["shear_stress"] = (
                    graph.ndata["shear_stress"] - self.shear_stress_min
                ) / (self.shear_stress_max - self.shear_stress_min)

            # Normalize edge data using Min-Max scaling
            if "x" in graph.edata and self.normalize.get("x", True):
                graph.edata["x"] = (
                    graph.edata["x"] - self.edge_x_min
                ) / (self.edge_x_max - self.edge_x_min)

            normalized_partitions.append(graph)
            
        # print("Max and Min values for coordinates:")
        # print("Max:", graph.ndata["coordinates"].max())
        # print("Min:", graph.ndata["coordinates"].min())

        # print("Max and Min values for normals:")
        # print("Max:", graph.ndata["normals"].max())
        # print("Min:", graph.ndata["normals"].min())

        # print("Max and Min values for area:")
        # print("Max:", graph.ndata["area"].max())
        # print("Min:", graph.ndata["area"].min())

        # print("Max and Min values for pressure:")
        # print("Max:", graph.ndata["pressure"].max())
        # print("Min:", graph.ndata["pressure"].min())

        # print("Max and Min values for shear_stress:")
        # print("Max:", graph.ndata["shear_stress"].max())
        # print("Min:", graph.ndata["shear_stress"].min())

        # print("Max and Min values for edge x:")
        # print("Max:", graph.edata["x"].max())
        # print("Min:", graph.edata["x"].min())
        
        return normalized_partitions, run_id


def create_dataloader_ChooseNorm(
    file_list,
    min_vals,
    max_vals,
    normalize,
    batch_size=1,
    shuffle=False,
    use_ddp=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
):
    """
    Creates a DataLoader for the GraphDataset with prefetching.

    Args:
        file_list (list of str): List of paths to .bin files.
        min_vals (np.ndarray): Global minimum values for normalization.
        max_vals (np.ndarray): Global maximum values for normalization.
        normalize (dict): Dictionary specifying which data to normalize.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for data loading.
        pin_memory (bool): If True, the data loader will copy tensors into CUDA pinned memory.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    dataset = GraphDataset(file_list, min_vals, max_vals, normalize)
    dataloader = GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        use_ddp=use_ddp,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    return dataloader


if __name__ == "__main__":
    data_path = "partitions"
    stats_file = "global_stats.json"

    # Load global statistics
    with open(stats_file, "r") as f:
        stats = json.load(f)
    min_vals = stats["min"]
    max_vals = stats["max"]

    # Find all .bin files in the directory
    file_list = find_bin_files(data_path)

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
    dataloader = create_dataloader_ChooseNorm(
        file_list,
        min_vals,
        max_vals,
        normalize,
        batch_size=1,
        prefetch_factor=None,
        use_ddp=False,
        num_workers=1,
    )

    # Example usage
    for batch_partitions, label in dataloader:
        for graph in batch_partitions:
            print(graph)
        print(label)