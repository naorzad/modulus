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
    """
    Compute global stats from processed file data.
    
    Args:
        results (list): List of feature data dictionaries from each file.
        mean_std_fields (list, optional): Fields to compute mean/std for; others use median/IQR.
        q_low (float): Lower percentile for IQR (default 25).
        q_high (float): Upper percentile for IQR (default 75).
    
    Returns:
        dict: Stats with 'mean', 'std', 'median', 'iqr' for each field.
    """
    results = [r for r in results if r is not None]
    if not results:
        raise ValueError("No valid data processed from .bin files.")
    
    all_data = {field: np.concatenate([r[field] for r in results], axis=0) 
                for field in results[0].keys()}
    
    stats = {"mean": {}, "std": {}, "median": {}, "iqr": {}, "skewness":{}}
    mean_std_fields = mean_std_fields or []

    for field, values in all_data.items():
        stats["mean"][field] = np.mean(values, axis=0).tolist()
        stats["std"][field] = np.std(values, axis=0).tolist()
        
        # Apply log transform for median/IQR on pressure and shear_stress
        # if field in ["pressure", "shear_stress"]:
        #     log_values = np.sign(values) * np.log1p(np.abs(values))
        #     stats["median"][field] = np.median(log_values, axis=0).tolist()
        #     q1, q3 = np.percentile(log_values, [q_low, q_high], axis=0)
        # else:
        stats["median"][field] = np.median(values, axis=0).tolist()
        q1, q3 = np.percentile(values, [q_low, q_high], axis=0)
        stats["iqr"][field] = (q3 - q1).tolist()
        stats["skewness"][field] = skew(values, axis=0).tolist()
    
    return stats

def compute_global_stats(bin_files, num_workers=4, mean_std_fields=None, q_low=10, q_high=90):
    """Compute global stats across all .bin files in parallel."""
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_file, bin_files), 
                            total=len(bin_files), desc="Processing BIN Files"))
    return aggregate_results(results, mean_std_fields, q_low, q_high)

def save_stats_to_json(stats, output_file):
    """Save stats to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function to compute and save scaling stats."""
    data_path = to_absolute_path(cfg.partitions_path)
    output_file = to_absolute_path(cfg.stats_file)
    
    bin_files = find_bin_files(data_path)
    if not bin_files:
        raise FileNotFoundError(f"No .bin files found in {data_path}")
    
    print(f"Found {len(bin_files)} .bin files.")
    # Use mean/std for coordinates, normals, area, x; median/IQR for pressure, shear_stress
    mean_std_fields = ["coordinates", "normals", "area", "x"]
    stats = compute_global_stats(bin_files, num_workers=cfg.num_preprocess_workers, 
                                 mean_std_fields=mean_std_fields)
    save_stats_to_json(stats, output_file)
    
    print(f"Stats saved to {output_file}")
    print("Mean:", stats["mean"])
    print("Std:", stats["std"])
    print("Median:", stats["median"])
    print("IQR:", stats["iqr"])
    print("skewness:", stats["skewness"])

if __name__ == "__main__":
    main()