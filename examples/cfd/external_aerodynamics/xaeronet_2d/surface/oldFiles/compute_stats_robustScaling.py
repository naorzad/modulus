"""
Extracts scaling stats (mean/std and median/IQR) from .bin graph files for training data normalization.
Processes node (coordinates, normals, area, pressure, shear_stress) and edge (x) features in parallel.
Flexible to use mean/std for specified fields and median/IQR for others.
"""

import os
import json
import numpy as np
import dgl
import hydra
from scipy.stats import skew
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_histograms(data_dict, fields=["pressure", "shear_stress"], bins=100):
    """Compute histograms for specified fields across entire dataset."""
    histograms = {}
    for field in fields:
        if field in data_dict:
            values = data_dict[field].flatten()
            hist, bin_edges = np.histogram(values, bins=bins, density=True)
            histograms[field] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist()
            }
    return histograms

def plot_histogram(stats, output_dir):
    """Plot and save histograms for pressure and shear_stress."""
    os.makedirs(output_dir, exist_ok=True)
    
    for field, hist_data in stats["histograms"].items():
        plt.figure(figsize=(12, 8))
        plt.stairs(hist_data["counts"], hist_data["bin_edges"], fill=True)
        plt.title(f"Histogram of {field} (Full Dataset)")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{field}_histogram_full.png"), dpi=300)
        plt.close()
        
def plot_pressure_shear_scatter(all_data, output_dir):
    """Create a scatter plot of pressure vs shear_stress for entire dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    pressure = all_data["pressure"].flatten()
    shear_stress = all_data["shear_stress"].flatten()
    
    # Ensure pressure and shear_stress have the same length
    min_length = min(len(pressure), len(shear_stress))
    pressure = pressure[:min_length]
    shear_stress = shear_stress[:min_length]
    
    plt.figure(figsize=(12, 8))
    plt.scatter(pressure, shear_stress, alpha=0.05, s=1)
    plt.xlabel("Pressure")
    plt.ylabel("Shear Stress")
    plt.title(f"Pressure vs Shear Stress Scatter (Full Dataset, {min_length} points)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "pressure_vs_shear_scatter_full.png"), dpi=300)
    plt.close()
    
def find_bin_files(data_path):
    """Find all .bin files in the specified directory."""
    return [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".bin")]

def process_file(bin_file):
    """Extract feature data from a .bin file containing graph partitions."""
    try:
        graphs, _ = dgl.load_graphs(bin_file)
    except Exception as e:
        print(f"Error loading {bin_file}: {e}")
        return None

    fields = {
        "node": ["coordinates", "normals", "area", "pressure", "shear_stress"],
        "edge": ["x"]
    }
    data = {field: [] for field in fields["node"] + fields["edge"]}

    for graph in graphs:
        for field in fields["node"]:
            if field in graph.ndata:
                values = graph.ndata[field].numpy()
                data[field].append(values if values.ndim > 1 else values[:, np.newaxis])
        for field in fields["edge"]:
            if field in graph.edata:
                data[field].append(graph.edata[field].numpy())

    return {field: np.concatenate(data[field], axis=0) for field in data if data[field]}

def aggregate_results(results, mean_std_fields=None, q_low=25, q_high=75):
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No valid data processed from .bin files.")
    
    all_data = {field: np.concatenate([r[field] for r in results], axis=0) 
                for field in results[0].keys()}
    
    stats = {"mean": {}, "std": {}, "median": {}, "iqr": {}, "skewness": {}, "histograms": {}}
    mean_std_fields = mean_std_fields or []

    for field, values in all_data.items():
        stats["mean"][field] = np.mean(values, axis=0).tolist()
        stats["std"][field] = np.std(values, axis=0).tolist()
        stats["median"][field] = np.median(values, axis=0).tolist()
        q1, q3 = np.percentile(values, [q_low, q_high], axis=0)
        stats["iqr"][field] = (q3 - q1).tolist()
        stats["skewness"][field] = skew(values, axis=0).tolist()
    
    stats["histograms"] = compute_histograms(all_data)
    
    return stats, all_data

def compute_global_stats(bin_files, num_workers=4, mean_std_fields=None, q_low=10, q_high=90):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_file, bin_files), 
                            total=len(bin_files), desc="Processing BIN Files"))
    
    stats, all_data = aggregate_results(results, mean_std_fields, q_low, q_high)
    return stats, all_data

def save_stats_to_json(stats, output_file):
    """Save stats to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    data_path = to_absolute_path(cfg.partitions_path)
    output_file = to_absolute_path(cfg.stats_file)
    output_dir = os.path.dirname(output_file)
    
    bin_files = find_bin_files(data_path)
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {data_path}")
    
    print(f"Found {len(bin_files)} .bin files.")
    mean_std_fields = ["coordinates", "normals", "area", "x"]
    stats, all_data = compute_global_stats(bin_files, num_workers=cfg.num_preprocess_workers, 
                                         mean_std_fields=mean_std_fields)
    
    # plot_histogram(stats, output_dir)
    # plot_pressure_shear_scatter(all_data, output_dir)
    
    save_stats_to_json(stats, output_file)
    
    print(f"Stats and histograms saved to {output_file}")
    print(f"Visualizations for full dataset saved to {output_dir}")
    print("Mean:", stats["mean"])
    print("Std:", stats["std"])
    print("Median:", stats["median"])
    print("IQR:", stats["iqr"])
    print("Skewness:", stats["skewness"])

if __name__ == "__main__":
    main()