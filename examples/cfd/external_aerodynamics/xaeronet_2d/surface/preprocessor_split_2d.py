"""
This code processes mesh data from .vtp files to create partitioned
graphs for large scale training. It extracts surface vertices and relevant attributes such as pressure
and shear stress. Using nearest neighbors, the code interpolates these attributes
for a sampled boundary of points, and constructs a graph based on these points, with
node features like coordinates, normals, pressure, and shear stress, as well as edge
features representing relative displacement. The graph is partitioned into subgraphs,
and the partitions are saved. The code supports parallel processing to handle multiple
samples simultaneously, improving efficiency. Additionally, it provides an option to
save the point cloud of each graph for visualization purposes.
"""

import os
import vtk
import pyvista as pv
import numpy as np
import torch
import dgl
import hydra
import random

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
from dgl.data.utils import save_graphs
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors


def calculate_area(points):
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # distances[:, 1] and distances[:, 2] are the distances to the two nearest neighbors
    point_areas = 0.5 * (distances[:, 1] + distances[:, 2])
    
    return point_areas

def fetch_mesh_vertices(mesh):
    """Fetches the vertices of a mesh."""
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    vertices = [points.GetPoint(i)[:2] for i in range(num_points)]  # Only X and Y
    return vertices


def add_edge_features(graph):
    """
    Add relative displacement and displacement norm as edge features to the graph.
    The calculations are done using the 'coordinates' attribute in the
    node data of each graph. The resulting edge features are stored in the 'x' attribute
    in the edge data of each graph.

    This method will modify the graph in-place.

    Returns
    -------
    dgl.DGLGraph
        Graph with updated edge features.
    """

    pos = graph.ndata.get("coordinates")
    if pos is None:
        raise ValueError(
            "'coordinates' does not exist in the node data of one or more graphs."
        )

    row, col = graph.edges()
    row = row.long()
    col = col.long()

    disp = pos[row][:, :2] - pos[col][:, :2]  # Only consider X and Y
    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
    graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

    return graph

def extract_global_context(file_name):
    """Extracts global context parameters from the file name."""
    parts = file_name.split('_')
    M = float(parts[1])
    ReL = float(parts[3])
    AOA = float(parts[5].replace('.vtk', ''))
    
    M_avg = 0.525
    M_std = 0.325
    M_norm = (M-M_avg)/M_std
    
    ReL_avg = 6.3
    ReL_std = 2.6
    ReL_norm = (ReL-ReL_avg)/ReL_std
    
    AOA_avg = 5
    AOA_std = 5
    AOA_norm = (AOA-AOA_avg)/AOA_std
    
    norm_vec = [M_norm, ReL_norm, AOA_norm]
    unnorm_vec = [M, ReL, AOA]
    return norm_vec, unnorm_vec
# def extract_global_context(file_name):
#     """Extracts global context parameters from the file name."""
#     parts = file_name.split('_')
#     M = float(parts[1])
#     ReL = float(parts[3])
#     AOA = float(parts[5].replace('.vtk', ''))
    
#     # Define min and max values for normalization
#     M_min = 0.2
#     M_max = 0.85
#     ReL_min = 5.0
#     ReL_max = 7.6
#     AOA_min = 0
#     AOA_max = 10
    
#     # Apply min-max normalization
#     M_norm = (M - M_min) / (M_max - M_min)
#     ReL_norm = (ReL - ReL_min) / (ReL_max - ReL_min)
#     AOA_norm = (AOA - AOA_min) / (AOA_max - AOA_min)
    
#     norm_vec = [M_norm, ReL_norm, AOA_norm]
#     unnorm_vec = [M, ReL, AOA]
#     return norm_vec, unnorm_vec

# Define this function outside of any local scope so it can be pickled
def run_task(params):
    """Wrapper function to unpack arguments for process_run."""
    return process_run(*params)


def process_partition(graph, num_partitions, halo_hops):
    """
    Helper function to partition a single graph and include node and edge features.
    """
    # Perform the partitioning
    partitioned = dgl.metis_partition(
        graph, k=num_partitions, extra_cached_hops=halo_hops, reshuffle=True
    )

    # For each partition, restore node and edge features
    partition_list = []
    for _, subgraph in partitioned.items():
        subgraph.ndata["coordinates"] = graph.ndata["coordinates"][
            subgraph.ndata[dgl.NID]
        ]
        subgraph.ndata["normals"] = graph.ndata["normals"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["pressure"] = graph.ndata["pressure"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["shear_stress"] = graph.ndata["shear_stress"][subgraph.ndata[dgl.NID]]
        subgraph.ndata["area"] = graph.ndata["area"][subgraph.ndata[dgl.NID]]  # Add area to node data
        
        if "x" in graph.edata:
            subgraph.edata["x"] = graph.edata["x"][subgraph.edata[dgl.EID]]

        partition_list.append(subgraph)

    return partition_list


def read_vtk(file_path):
    """Reads a .vtk file and returns a PyVista mesh."""
    return pv.read(file_path)


def visualize_normals(points, normals):
    """Visualize 2D points and their normals using PyVista."""
    # Add a zero vector for the Z-coordinate
    points_3d = np.column_stack((points, np.zeros(points.shape[0])))
    
    # Create a PyVista PolyData object
    point_cloud = pv.PolyData(points_3d)
    
    # Add normals to the PolyData object
    normals_3d = np.column_stack((normals, np.zeros(normals.shape[0])))
    point_cloud["normals"] = normals_3d
    
    # Create a plotter object
    plotter = pv.Plotter()
    
    # Add the points to the plotter
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)
    
    # Add the normals to the plotter
    plotter.add_arrows(points_3d, normals_3d, mag=0.1, color='red')
    
    # Show the plot
    plotter.show()

def calculate_2d_centroid(points):
    """Calculate the centroid of a set of 2D points."""
    return np.mean(points, axis=0)

def ensure_normals_point_outward(points, normals):
    """Ensure normals point outward for a closed 2D shape."""
    centroid = calculate_2d_centroid(points)
    for i, point in enumerate(points):
        normal = normals[i]
        direction_to_centroid = centroid - point
        if np.dot(normal, direction_to_centroid) > 0:
            normals[i] = -normal  # Reverse the normal
    return normals

def calculate_2d_normals(points, n_neighbors=2, epsilon=1e-6):
    """Calculate normals for a 2D point cloud with weighted averaging and normalization."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(points)
    _, indices = nbrs.kneighbors(points)
    
    normals = np.zeros(points.shape, dtype=np.float32)
    
    for i, neighbors in enumerate(indices):
        p0 = points[i]
        normal_sum = np.zeros(2, dtype=np.float32)
        weight_sum = 0.0
        
        for j in range(1, n_neighbors):
            p1 = points[neighbors[j]]
            vector = p1 - p0
            distance = np.linalg.norm(vector)
            if distance > epsilon:
                normal = np.array([-vector[1], vector[0]], dtype=np.float32)  # Perpendicular vector
                normal /= distance  # Normalize
                weight = 1.0 / distance  # Weight inversely proportional to distance
                normal_sum += weight * normal
                weight_sum += weight
        
        if weight_sum > 0:
            normal_avg = normal_sum / weight_sum  # Weighted average
            norm = np.linalg.norm(normal_avg)
            if norm > epsilon:
                normals[i] = normal_avg / norm  # Normalize the final normal
    normals = ensure_normals_point_outward(points, normals)
    # visualize_normals(points, normals)
    return normals

def split_data(run_dirs, train_ratio=0.93, val_ratio=0.05, test_ratio=0.02):
    """Split the run directories into training, validation, and test sets."""
    random.shuffle(run_dirs)
    total_runs = len(run_dirs)
    train_end = int(total_runs * train_ratio)
    val_end = train_end + int(total_runs * val_ratio)
    
    train_dirs = run_dirs[:train_end]
    val_dirs = run_dirs[train_end:val_end]
    test_dirs = run_dirs[val_end:]
    
    return train_dirs, val_dirs, test_dirs

def save_partitioned_graphs(partitioned_graphs, partition_file_path):
    """Save partitioned graphs to the specified file path."""
    save_graphs(partition_file_path, partitioned_graphs)

def process_run(
    run_path, point_list, node_degree, num_partitions, halo_hops, save_point_cloud=False, data_type="train", file_index=1
):
    """Process a single run directory to generate a multi-level graph and apply partitioning."""
    run_id = os.path.basename(run_path).split("_")[-1]

    # Find the only .vtk file in the folder
    vtk_files = [f for f in os.listdir(run_path) if f.endswith(".vtk")]
    if len(vtk_files) != 1:
        print(f"Warning: Expected one .vtk file in folder {run_path}, found {len(vtk_files)}. Skipping...")
        return

    vtk_file = os.path.join(run_path, vtk_files[0])
    
    # Extract global context from file name
    global_context, global_context_unnorm = extract_global_context(vtk_files[0])
    
    # Path to save the list of partitions and global context
    partition_file_path = to_absolute_path(f"/workspace/NACA0012_SurfaceFlow/{data_type}_partitions/graph_partitions_{file_index}.bin")
    context_file_path = os.path.join(os.path.dirname(partition_file_path), f"global_context_{file_index}.npy")
    context_file_path_unnorm = os.path.join(os.path.dirname(partition_file_path), f"global_context_{file_index}_unnorm.npy")

    if os.path.exists(partition_file_path):
        print(f"Partitions for run {run_id} already exist. Skipping...")
        return

    if not os.path.exists(vtk_file):
        print(f"Warning: Missing files for run {run_id}. Skipping...")
        return

    try:
        # Load the VTK file
        surface_mesh = read_vtk(vtk_file)
        surface_vertices = fetch_mesh_vertices(surface_mesh)
        surface_mesh = surface_mesh.cell_data_to_point_data()
        node_attributes = surface_mesh.point_data
        pressure_ref = node_attributes["Pressure_Coefficient"]
        shear_stress_ref = node_attributes["Skin_Friction_Coefficient"]

        # Use the exact points from the .vtk file
        all_points = np.array(surface_vertices)
        all_normals = calculate_2d_normals(all_points)  # Calculate normals
        all_areas = calculate_area(all_points)  # Calculate areas
        edge_sources = []
        edge_destinations = []

        # Construct edges for the point cloud
        nbrs_points = NearestNeighbors(
            n_neighbors=node_degree + 1, algorithm="ball_tree"
        ).fit(all_points)
        _, indices_within = nbrs_points.kneighbors(all_points)
        src_within = [i for i in range(len(all_points)) for _ in range(node_degree)]
        dst_within = indices_within[:, 1:].flatten()

        # Add the within-level edges
        edge_sources.extend(src_within)
        edge_destinations.extend(dst_within)

        # Compute pressure and shear stress for the point cloud
        nbrs_surface = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(
            surface_vertices
        )
        _, indices = nbrs_surface.kneighbors(all_points)
        indices = indices.flatten()

        pressure = pressure_ref[indices]
        shear_stress = shear_stress_ref[indices]

    except Exception as e:
        print(f"Error processing run {run_id}: {e}. Skipping this run...")
        return

    try:
        # Create the final graph with multi-level edges
        graph = dgl.graph((edge_sources, edge_destinations))
        graph = dgl.remove_self_loop(graph)
        graph = dgl.to_simple(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)

        graph.ndata["coordinates"] = torch.tensor(all_points, dtype=torch.float32)  # Only X and Y
        graph.ndata["normals"] = torch.tensor(all_normals, dtype=torch.float32)
        graph.ndata["pressure"] = torch.tensor(pressure, dtype=torch.float32).unsqueeze(-1)
        graph.ndata["shear_stress"] = torch.tensor(shear_stress[:, :2], dtype=torch.float32)
        graph.ndata["area"] = torch.tensor(all_areas, dtype=torch.float32).unsqueeze(-1)  # Add area to node data
        graph = add_edge_features(graph)

        # Partition the graph
        partitioned_graphs = process_partition(graph, num_partitions, halo_hops)

        # Save the partitions and global context
        save_partitioned_graphs(partitioned_graphs, partition_file_path)
        np.save(context_file_path, global_context)
        np.save(context_file_path_unnorm, global_context_unnorm)
        
        if save_point_cloud:
            point_cloud_dir = f"/workspace/NACA0012_SurfaceFlow/{data_type}_point_clouds"
            os.makedirs(point_cloud_dir, exist_ok=True)
            point_cloud_path = os.path.join(point_cloud_dir, f"point_cloud_{file_index}.vtp")
            point_cloud = pv.PolyData(np.column_stack((graph.ndata["coordinates"].numpy(), np.zeros((graph.ndata["coordinates"].shape[0], 1)))))
            point_cloud["coordinates"] = np.column_stack((graph.ndata["coordinates"].numpy(), np.zeros((graph.ndata["coordinates"].shape[0], 1))))
            point_cloud["normals"] = np.column_stack((graph.ndata["normals"].numpy(), np.zeros((graph.ndata["normals"].shape[0], 1))))
            point_cloud["pressure"] = graph.ndata["pressure"].numpy()  # Keep pressure as it is
            point_cloud["shear_stress"] = graph.ndata["shear_stress"].numpy()
            point_cloud["area"] = graph.ndata["area"].numpy()  # Add area to point cloud
            # point_cloud["x"] = graph.edata["x"].numpy()  # Add area to point cloud
            point_cloud.save(point_cloud_path)

    except Exception as e:
        print(
            f"Error while constructing graph or saving data for run {run_id}: {e}. Skipping this run..."
        )
        return

def process_all_runs(
    base_path,
    num_points,
    node_degree,
    num_partitions,
    halo_hops,
    num_workers=8,
    save_point_cloud=False,
):
    """Process all runs in the base directory in parallel."""

    run_dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if d.startswith("run_") and os.path.isdir(os.path.join(base_path, d))
    ]

    train_dirs, val_dirs, test_dirs = split_data(run_dirs)

    tasks_train = [
        (run_dir, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "train", i+1)
        for i, run_dir in enumerate(train_dirs)
    ]

    tasks_val = [
        (run_dir, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "validation", i+1)
        for i, run_dir in enumerate(val_dirs)
    ]

    tasks_test = [
        (run_dir, num_points, node_degree, num_partitions, halo_hops, save_point_cloud, "test", i+1)
        for i, run_dir in enumerate(test_dirs)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for _ in tqdm(
            pool.map(run_task, tasks_train),
            total=len(tasks_train),
            desc="Processing Training Runs",
            unit="run",
        ):
            pass

        for _ in tqdm(
            pool.map(run_task, tasks_val),
            total=len(tasks_val),
            desc="Processing Validation Runs",
            unit="run",
        ):
            pass

        for _ in tqdm(
            pool.map(run_task, tasks_test),
            total=len(tasks_test),
            desc="Processing Test Runs",
            unit="run",
        ):
            pass

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    process_all_runs(
        base_path=to_absolute_path(cfg.data_path),
        num_points=cfg.num_nodes,
        node_degree=cfg.node_degree,
        num_partitions=cfg.num_partitions,
        halo_hops=cfg.num_message_passing_layers,
        num_workers=cfg.num_preprocess_workers,
        save_point_cloud=cfg.save_point_clouds,
    )

if __name__ == "__main__":
    main()