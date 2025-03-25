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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import dgl
import pandas as pd
import torch
import yaml
from dgl.data import DGLDataset
from torch import Tensor
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

try:
    import pyvista as pv
    import vtk
except ImportError:
    raise ImportError(
        "naca0012 Dataset requires the vtk and pyvista libraries. "
        "Install with pip install vtk pyvista"
    )

@dataclass
class FileInfo:
    """VTP file info storage."""

    mach: float
    reynolds_number: float
    AoA: float
    
@dataclass
class MetaData(DatapipeMetaData):
    name: str = "naca0012"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class naca0012Dataset(DGLDataset, Datapipe):
    """
    DrivAerNet dataset.

    Note: DrivAerNetDataset does not use default DGLDataset caching
    functionality such as `has_cache`, `download` etc,
    as it is invoked during the __init__ call so takes a lot of time.
    Instead, DrivAerNetDataset caches graphs in __getitem__ call thus
    avoiding long initialization delay.

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    coeff_filename: str, optional
        DrivAerNet coefficients file name, default is from the dataset location.
    invar_keys: Iterable[str], optional
        The input node features to consider. Default includes 'pos'.
    outvar_keys: Iterable[str], optional
        The output features to consider. Default includes 'p' and 'wallShearStress'.
    normalize_keys Iterable[str], optional
        The features to normalize. Default includes 'p' and 'wallShearStress'.
    cache_dir: str, optional
        Path to the cache directory to store graphs in DGL format for fast loading.
        Default is ./cache/.
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        num_samples: int = 10,
        # coeff_filename: str = "AeroCoefficients_DrivAerNet_FilteredCorrected.csv",
        invar_keys: Iterable[str] = (
            "pos",
            "mach",
            "reynolds_number",
            "AoA",
        ),
        outvar_keys: Iterable[str] = ("p", "wallShearStress"),
        normalize_keys: Iterable[str] = (
            "p",
            "wallShearStress",
            "mach",
            "reynolds_number",
            "AoA",
        ),
        normalization_bound: Tuple[float, float] = (-1.0, 1.0),
        cache_dir: str | Path = "./cache/",
        force_reload: bool = False,
        name: str = "dataset",
        compute_drag: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        DGLDataset.__init__(
            self,
            name=name,
            force_reload=force_reload,
            verbose=verbose
        )
        Datapipe.__init__(
            self,
            meta=MetaData()
        )

        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(
                f"Path {self.data_dir} does not exist or is not a directory."
            )

        self.split = split.lower()
        if split not in (splits := ["train", "val", "test"]):
            raise ValueError(f"{split = } is not supported, must be one of {splits}.")

        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)
        self.normalization_bound = normalization_bound
        self.compute_drag = compute_drag

        self.cache_dir = (
            self._get_cache_dir(self.data_dir, Path(cache_dir))
            if cache_dir is not None
            else None
        )

        # Get case ids from the list of .vtp files.
        case_files = []
        case_info_files = []
        self.case_ids = []
        for case_file in sorted(self.data_dir.glob("*.vtp")):
            case_id = int(str(case_file.stem).removeprefix("case"))
            # Check if there is a corresponding info file.
            case_info_file = self.info_dir / f"case{case_id}_info.txt"
            if not case_info_file.is_file():
                raise IOError(f"File not found {case_info_file}")
            case_files.append(str(case_file))
            case_info_files.append(str(case_info_file))
            self.case_ids.append(case_id)

        self.graphs = [None] * self.length
        if self.compute_drag:
            self.normals = [None] * self.length
            self.areas = [None] * self.length
            self.coeff = [None] * self.length
            
        # TODO(akamenev): these are estimates from small sample, need to compute from full data.
        self.nstats = {
            k: {"mean": v[0], "std": v[1]}
            for k, v in {
                "p": (-94.50448, 117.25317),
                "wallShearStress": (
                    torch.tensor([-0.56926626, 0.0027714, -0.07354721]),
                    torch.tensor([0.82198745, 0.45956784, 0.7490267]),
                ),
            }.items()
        }

        self.estats = {
            "x": {
                "mean": torch.tensor([0, 0, 0, 0.01338306]),
                "std": torch.tensor([0.00512953, 0.00953013, 0.00923065, 0.00482016]),
            }
        }
        

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        if not 0 <= idx < len(self):
            raise IndexError(f"Invalid {idx = }, must be in [0, {len(self)})")

        coeffs = self.coeffs.iloc[idx]
        gname = coeffs.name

        if self.cache_dir is None:
            # Caching is disabled - create the graph.
            graph = self._create_dgl_graph(gname)
        else:
            cached_graph_filename = self.cache_dir / (gname + ".bin")
            if not self._force_reload and cached_graph_filename.is_file():
                gs, _ = dgl.load_graphs(str(cached_graph_filename))
                if len(gs) != 1:
                    raise ValueError(f"Expected to load 1 graph but got {len(gs)}.")
                graph = gs[0]
            else:
                graph = self._create_dgl_graph(gname)
                dgl.save_graphs(str(cached_graph_filename), [graph])

        # Set graph inputs/outputs.
        graph.ndata["x"] = torch.cat([graph.ndata[k] for k in self.input_keys], dim=-1)
        graph.ndata["y"] = torch.cat([graph.ndata[k] for k in self.output_keys], dim=-1)

        return {
            "name": gname,
            "graph": graph,
            "c_d": torch.tensor(coeffs["Average Cd"], dtype=torch.float32),
        }

    @staticmethod
    def _get_cache_dir(data_dir, cache_dir):
        if not cache_dir.is_absolute():
            cache_dir = data_dir / cache_dir
        return cache_dir.resolve()

    def _create_dgl_graph(
        self,
        name: str,
        to_bidirected: bool = True,
        dtype: torch.dtype | str = torch.int32,
    ) -> dgl.DGLGraph:
        """Creates a DGL graph from DrivAerNet VTP data.

        Parameters
        ----------
        name : str
            Name of the graph in DrivAerNet.
        to_bidirected : bool, optional
            Whether to make the graph bidirected. Default is True.
        dtype : torch.dtype or str, optional
            Data type for the graph. Default is torch.int32.

        Returns
        -------
        dgl.DGLGraph
            The DGL graph.
        """

        def extract_edges(mesh: pv.PolyData) -> list[tuple[int, int]]:
            # Extract connectivity information from the mesh.
            # Traversal API is faster comparing to iterating over mesh.cell.
            polys = mesh.GetPolys()
            if polys is None:
                raise ValueError("Failed to get polygons from the mesh.")

            polys.InitTraversal()

            edge_list = []
            for _ in range(polys.GetNumberOfCells()):
                id_list = vtk.vtkIdList()
                polys.GetNextCell(id_list)
                num_ids = id_list.GetNumberOfIds()
                for j in range(num_ids - 1):
                    edge_list.append(  # noqa: PERF401
                        (id_list.GetId(j), id_list.GetId(j + 1))
                    )
                # Add the final edge between the last and the first vertices.
                edge_list.append((id_list.GetId(num_ids - 1), id_list.GetId(0)))

            return edge_list

        def permute_mesh(p_vtk_path: Path, wss_vtk_path: Path) -> Tensor:
            # The issue with DrivAerNet dataset is pressure and WSS meshes
            # are stored in different files. Even though each file contains
            # the same mesh coordinates, the nodes are permuted (order does not match)
            # which makes it impossible to do simple point_data assignment.
            # This method permutes WSS mesh by using vtkProbeFilter.

            p_reader = vtk.vtkPolyDataReader()
            p_reader.SetFileName(p_vtk_path)
            p_reader.Update()
            p_out = p_reader.GetOutput()

            wss_reader = vtk.vtkPolyDataReader()
            wss_reader.SetFileName(wss_vtk_path)
            wss_reader.Update()
            wss_out = wss_reader.GetOutput()

            probe = vtk.vtkProbeFilter()
            # p mesh is the input for which corresponding values from
            # wss mesh are retrieved.
            probe.SetInputData(p_out)
            probe.SetSourceData(wss_out)
            probe.Update()

            probe_out = probe.GetOutput()
            wss_arr = probe_out.GetPointData().GetArray("wallShearStress")
            num_points = p_out.GetNumberOfPoints()
            wss = torch.empty((num_points, 3), dtype=torch.float32)
            for i in range(num_points):
                x, y, z = wss_arr.GetTuple3(i)
                wss[i, 0] = x
                wss[i, 1] = y
                wss[i, 2] = z

            return wss

        # Load the pressure mesh even if p is not selected.
        # The p and wss meshes contain the same mesh nodes,
        # so use nodes from p for simplicity.
        p_vtk_path = self.p_vtk_dir / (name + ".vtk")
        p_mesh = pv.read(p_vtk_path)

        edge_list = extract_edges(p_mesh)

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(p_mesh.points, dtype=torch.float32)

        if (k := "p") in self.output_keys:
            graph.ndata[k] = torch.tensor(p_mesh.point_data[k], dtype=torch.float32)

        if (k := "wallShearStress") in self.output_keys:
            wss_vtk_path = self.wss_vtk_dir / (name + ".vtk")
            graph.ndata[k] = permute_mesh(p_vtk_path, wss_vtk_path)

        # Normalize nodes.
        for k in self.input_keys + self.output_keys:
            if k not in self.normalize_keys:
                continue
            v = (graph.ndata[k] - self.nstats[k]["mean"]) / self.nstats[k]["std"]
            graph.ndata[k] = v.unsqueeze(-1) if v.ndim == 1 else v

        # Add edge features which contain relative edge nodes displacement and
        # displacement norm. Stored as `x` in the graph edge data.
        u, v = graph.edges()
        pos = graph.ndata["pos"]
        disp = pos[u] - pos[v]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        # Normalize edges.
        for k, v in graph.edata.items():
            v = (v - self.estats[k]["mean"]) / self.estats[k]["std"]
            graph.edata[k] = v.unsqueeze(-1) if v.ndim == 1 else v

        return graph

    @torch.no_grad
    def denormalize(
        self, pred: Tensor, gt: Tensor, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Denormalizes the inputs using previously collected statistics."""

        def denorm(x: Tensor, name: str):
            stats = self.nstats[name]
            mean = torch.as_tensor(stats["mean"]).to(device)
            std = torch.as_tensor(stats["std"]).to(device)
            return x * std + mean

        pred_d = []
        gt_d = []
        pred_d.append(denorm(pred[:, :1], "p"))
        gt_d.append(denorm(gt[:, :1], "p"))

        if (k := "wallShearStress") in self.output_keys:
            pred_d.append(denorm(pred[:, 1:4], k))
            gt_d.append(denorm(gt[:, 1:4], k))

        return torch.cat(pred_d, dim=-1), torch.cat(gt_d, dim=-1)
