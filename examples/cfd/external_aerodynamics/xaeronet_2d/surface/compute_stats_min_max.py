import os
import json
import numpy as np
import dgl
import hydra

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

def find_bin_files(data_path):
    """
    Finds all .bin files in the specified directory.
    """
    return [
        os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".bin")
    ]

def process_file(bin_file):
    """
    Processes a single .bin file containing graph partitions to compute the min, max, mean, mean of squares, and count for each variable.
    """
    graphs, _ = dgl.load_graphs(bin_file)

    # Initialize dictionaries to accumulate stats
    node_fields = ["coordinates", "normals", "area", "pressure", "shear_stress"]
    edge_fields = ["x"]

    field_mins = {}
    field_maxs = {}
    field_means = {}
    field_square_means = {}
    counts = {}

    # Initialize stats accumulation for each partitioned graph
    for field in node_fields + edge_fields:
        field_mins[field] = np.inf
        field_maxs[field] = -np.inf
        field_means[field] = 0
        field_square_means[field] = 0
        counts[field] = 0

    # Loop through each partition in the file
    for graph in graphs:
        # Process node data
        for field in node_fields:
            if field in graph.ndata:
                data = graph.ndata[field].numpy()

                if data.ndim == 1:
                    data = np.expand_dims(data, axis=-1)

                # Compute min, max, mean, mean of squares, and count for each partition
                field_min = np.min(data, axis=0)
                field_max = np.max(data, axis=0)
                field_mean = np.mean(data, axis=0)
                field_square_mean = np.mean(data**2, axis=0)
                count = data.shape[0]

                # Accumulate stats across partitions
                field_mins[field] = np.minimum(field_mins[field], field_min)
                field_maxs[field] = np.maximum(field_maxs[field], field_max)
                field_means[field] += field_mean * count
                field_square_means[field] += field_square_mean * count
                counts[field] += count
            else:
                print(f"Warning: Node field '{field}' not found in {bin_file}")

        # Process edge data
        for field in edge_fields:
            if field in graph.edata:
                data = graph.edata[field].numpy()

                field_min = np.min(data, axis=0)
                field_max = np.max(data, axis=0)
                field_mean = np.mean(data, axis=0)
                field_square_mean = np.mean(data**2, axis=0)
                count = data.shape[0]

                # Accumulate stats across partitions
                field_mins[field] = np.minimum(field_mins[field], field_min)
                field_maxs[field] = np.maximum(field_maxs[field], field_max)
                field_means[field] += field_mean * count
                field_square_means[field] += field_square_mean * count
                counts[field] += count
            else:
                print(f"Warning: Edge field '{field}' not found in {bin_file}")

    return field_mins, field_maxs, field_means, field_square_means, counts

def aggregate_results(results, epsilon=1e-6):
    """
    Aggregates the results from all files to compute global min, max, mean and standard deviation.
    """
    total_min = {}
    total_max = {}
    total_mean = {}
    total_square_mean = {}
    total_count = {}

    # Initialize totals with extreme values for each field
    for field in results[0][0].keys():
        total_min[field] = np.inf
        total_max[field] = -np.inf
        total_mean[field] = 0
        total_square_mean[field] = 0
        total_count[field] = 0

    # Accumulate min, max values and weighted sums and counts for mean and std dev.
    for (field_mins, field_maxs, field_means, field_square_means, counts) in results:
        for field in field_mins:
            total_min[field] = np.minimum(total_min[field], field_mins[field])
            total_max[field] = np.maximum(total_max[field], field_maxs[field])
            total_mean[field] += field_means[field]
            total_square_mean[field] += field_square_means[field]
            total_count[field] += counts[field]

    # Compute global mean and standard deviation
    global_mean = {}
    global_std_dev = {}

    for field in total_mean:
        global_mean[field] = total_mean[field] / total_count[field]
        variance = (total_square_mean[field] / total_count[field]) - (
            global_mean[field] ** 2
        )
        global_std_dev[field] = np.sqrt(
            np.maximum(variance, 0)
        )  # Ensure no negative variance due to rounding errors

        # Replace zero standard deviation with epsilon
        global_std_dev[field][global_std_dev[field] < epsilon] = epsilon

    return total_min, total_max, global_mean, global_std_dev

def compute_global_stats(bin_files, num_workers=4, epsilon=1e-6):
    """
    Computes the global min, max, mean and standard deviation for each field across all .bin files using parallel processing.
    """
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_file, bin_files),
                total=len(bin_files),
                desc="Processing BIN Files",
                unit="file",
            )
        )

    # Aggregate the results from all files
    global_min, global_max, global_mean, global_std_dev = aggregate_results(results)

    return global_min, global_max, global_mean, global_std_dev

def save_stats_to_json(min_vals, max_vals, mean_vals, std_dev_vals, output_file):
    """
    Saves the global min, max, mean, and standard deviation to a JSON file.
    """
    stats = {
        "min": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in min_vals.items()
        },
        "max": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in max_vals.items()
        },
        "mean": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in mean_vals.items()
        },
        "std_dev": {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in std_dev_vals.items()
        },
    }

    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = to_absolute_path(cfg.partitions_path)  # Directory containing the .bin graph files with partitions
    output_file = to_absolute_path(cfg.stats_file)  # File to save the global statistics

    # Find all .bin files in the directory
    bin_files = find_bin_files(data_path)

    # Compute global statistics with parallel processing
    global_min, global_max, global_mean, global_std_dev = compute_global_stats(
        bin_files, num_workers=cfg.num_preprocess_workers, epsilon=cfg.epsilon
    )

    # Save statistics to a JSON file
    save_stats_to_json(global_min, global_max, global_mean, global_std_dev, output_file)

    # Print the results
    print("Global Min:", global_min)
    print("Global Max:", global_max)
    print("Global Mean:", global_mean)
    print("Global Standard Deviation:", global_std_dev)
    print(f"Statistics saved to {output_file}")

if __name__ == "__main__":
    main()