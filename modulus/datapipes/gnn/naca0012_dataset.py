from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import dgl
import torch
from dgl.data import DGLDataset
from torch import Tensor

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
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        num_samples: int = 10,
        invar_keys: Iterable[str] = ("pos", "mach", "reynolds_number", "AoA"),
        outvar_keys: Iterable[str] = ("pressure_coefficient", "skin_friction_coefficient"),
        normalize_keys: Iterable[str] = ("mach", "reynolds_number", "AoA"),
        normalization_bound: Tuple[float, float] = (-1.0, 1.0),
        cache_dir: str | Path = "./cache/",
        force_reload: bool = False,
        name: str = "dataset",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        DGLDataset.__init__(self, name=name, force_reload=force_reload, verbose=verbose)
        Datapipe.__init__(self, meta=MetaData())

        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f"Path {self.data_dir} does not exist or is not a directory.")

        self.split = split.lower()
        if split not in (splits := ["train", "val", "test"]):
            raise ValueError(f"{split = } is not supported, must be one of {splits}.")

        self.input_keys = list(invar_keys)
        self.output_keys = list(outvar_keys)
        self.normalize_keys = list(normalize_keys)
        self.normalization_bound = normalization_bound

        self.cache_dir = (
            self._get_cache_dir(self.data_dir, Path(cache_dir))
            if cache_dir is not None
            else None
        )

        # Get case ids from the list of .vtk files.
        self.case_files = sorted(self.data_dir.glob("*.vtk"))

        if len(self.case_files) < num_samples:
            raise ValueError(f"Number of available {self.split} dataset entries ({len(self.case_files)}) is less than the number of samples ({num_samples})")

        self.case_files = self.case_files[:num_samples]

        # TODO: Update normalization statistics based on your dataset
        self.nstats = {
            k: {"mean": v[0], "std": v[1]}
            for k, v in {
                "mach": (0.2, 1.0),
                "reynolds_number": (5.0, 7.5),
                "AoA": (0.0, 10.0),
            }.items()
        }

        self.estats = {
            "x": {
                "mean": torch.tensor([0, 0, 0, 0.01338306]),
                "std": torch.tensor([0.00512953, 0.00953013, 0.00923065, 0.00482016]),
            }
        }

    def __len__(self) -> int:
        return len(self.case_files)

    def __getitem__(self, idx: int) -> dgl.DGLGraph:
        if not 0 <= idx < len(self):
            raise IndexError(f"Invalid {idx = }, must be in [0, {len(self)})")

        case_file = self.case_files[idx]
        gname = case_file.stem

        if self.cache_dir is None:
            # Caching is disabled - create the graph.
            graph = self._create_dgl_graph(gname, case_file)
        else:
            cached_graph_filename = self.cache_dir / (gname + ".bin")
            if not self._force_reload and cached_graph_filename.is_file():
                gs, _ = dgl.load_graphs(str(cached_graph_filename))
                if len(gs) != 1:
                    raise ValueError(f"Expected to load 1 graph but got {len(gs)}.")
                graph = gs[0]
            else:
                graph = self._create_dgl_graph(gname, case_file)
                dgl.save_graphs(str(cached_graph_filename), [graph])

        # Set graph inputs/outputs.
        graph.ndata["x"] = torch.cat([graph.ndata[k] for k in self.input_keys], dim=-1)
        graph.ndata["y"] = torch.cat([graph.ndata[k] for k in self.output_keys], dim=-1)

        return {
            "name": gname,
            "graph": graph,
        }

    @staticmethod
    def _get_cache_dir(data_dir, cache_dir):
        if not cache_dir.is_absolute():
            cache_dir = data_dir / cache_dir
        return cache_dir.resolve()

    def _create_dgl_graph(
        self,
        name: str,
        case_file: Path,
        to_bidirected: bool = True,
        dtype: torch.dtype | str = torch.int32,
    ) -> dgl.DGLGraph:
        """Creates a DGL graph from NACA 0012 VTK data."""

        def extract_edges(mesh: pv.PolyData) -> list[tuple[int, int]]:
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
                    edge_list.append((id_list.GetId(j), id_list.GetId(j + 1)))
                edge_list.append((id_list.GetId(num_ids - 1), id_list.GetId(0)))

            return edge_list

        # Extract Mach, Reynolds number, and AoA from the file name
        parts = name.split('_')
        mach = float(parts[1])
        reynolds_number = float(parts[3])
        AoA = float(parts[5])

        # Load the VTK file
        mesh = pv.read(case_file)

        edge_list = extract_edges(mesh)

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(mesh.points, dtype=torch.float32)
        graph.ndata["mach"] = torch.full((mesh.number_of_points, 1), mach, dtype=torch.float32)
        graph.ndata["reynolds_number"] = torch.full((mesh.number_of_points, 1), reynolds_number, dtype=torch.float32)
        graph.ndata["AoA"] = torch.full((mesh.number_of_points, 1), AoA, dtype=torch.float32)

        if (k := "pressure_coefficient") in self.output_keys:
            graph.ndata[k] = torch.tensor(mesh.point_data[k], dtype=torch.float32)

        if (k := "skin_friction_coefficient") in self.output_keys:
            graph.ndata[k] = torch.tensor(mesh.point_data[k], dtype=torch.float32)

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
    ) -> Tuple[Tensor, Tensor]:
        """Denormalizes the inputs using previously collected statistics."""

        pred_d = []
        gt_d = []
        pred_d.append(pred[:, :1])  # pressure_coefficient
        gt_d.append(gt[:, :1])  # pressure_coefficient

        if "skin_friction_coefficient" in self.output_keys:
            pred_d.append(pred[:, 1:4])  # skin_friction_coefficient
            gt_d.append(gt[:, 1:4])  # skin_friction_coefficient

        return torch.cat(pred_d, dim=-1), torch.cat(gt_d, dim=-1)