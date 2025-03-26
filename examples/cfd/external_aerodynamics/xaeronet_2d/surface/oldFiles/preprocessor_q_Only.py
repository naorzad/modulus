"""
Standalone script to read global context variables from preprocessed .npy files,
calculate dynamic pressure (q) properties (min, max, mean, std), and update the
normalized context files with q normalized using mean and std.
"""

import os
import numpy as np
import math
import json
from tqdm import tqdm

# Configuration (adjust these paths as needed)
BASE_PATH = "/workspace/NACA0012_SurfaceFlow"
DATA_TYPES = ["train_partitions", "validation_partitions", "test_partitions"]
Q_STATS_FILE = "conf/q_stats.json"

def calculate_dynamic_pressure(M, Re):
    # Constants
    gamma = 1.4  # Ratio of specific heats
    R = 287.87  # Specific gas constant in J/kg·K
    T = 273.15  # Free-stream temperature in K
    L = 1.0  # Reynolds length in meters
    mu = 1.853E-5  # Free-stream viscosity in kg/(m·s)

    # Calculate the speed of sound
    a = math.sqrt(gamma * R * T)

    # Calculate free-stream velocity
    U = M * a

    # Calculate fluid density
    rho = (Re * mu) / (U * L)

    # Calculate dynamic pressure
    q = 0.5 * rho * U**2
    return q

def get_context_files(base_path, data_types):
    """Collect all context file paths across data types."""
    context_files = []
    unnorm_context_files = []
    for data_type in data_types:
        dir_path = os.path.join(base_path, data_type)
        if not os.path.exists(dir_path):
            print(f"Warning: Directory {dir_path} does not exist. Skipping...")
            continue
        for file in os.listdir(dir_path):
            if file.startswith("global_context_") and file.endswith(".npy") and "_unnorm" not in file:
                index = file.split("_")[-1].replace(".npy", "")
                context_file = os.path.join(dir_path, file)
                unnorm_file = os.path.join(dir_path, f"global_context_{index}_unnorm.npy")
                if os.path.exists(unnorm_file):
                    context_files.append(context_file)
                    unnorm_context_files.append(unnorm_file)
                else:
                    print(f"Warning: Unnormalized context file {unnorm_file} missing. Skipping {context_file}...")
    return context_files, unnorm_context_files

def compute_q_stats(context_files, unnorm_context_files):
    """Compute q statistics from unnormalized context data."""
    q_values = []
    
    for context_file, unnorm_file in tqdm(zip(context_files, unnorm_context_files), total=len(context_files), desc="Computing q values"):
        unnorm_context = np.load(unnorm_file)
        if len(unnorm_context) < 3:
            print(f"Warning: {unnorm_file} has insufficient data ({len(unnorm_context)} elements). Skipping...")
            continue
        M, ReL, _ = unnorm_context[:3]  # Extract M, ReL, AOA
        q = calculate_dynamic_pressure(M, 10**ReL)
        q_values.append(q)
    
    if not q_values:
        raise ValueError("No valid q values computed. Check context files.")

    q_array = np.array(q_values)
    q_stats = {
        "min": float(np.min(q_array)),
        "max": float(np.max(q_array)),
        "mean": float(np.mean(q_array)),
        "std": float(np.std(q_array))
    }
    return q_stats, q_values

def update_context_files(context_files, unnorm_context_files, q_values, q_stats):
    """Update normalized context files with q_norm."""
    for context_file, unnorm_file, q in tqdm(zip(context_files, unnorm_context_files, q_values), total=len(context_files), desc="Updating context files"):
        context = np.load(context_file)
        unnorm_context = np.load(unnorm_file)
        
        # Normalize q
        q_norm = (q - q_stats["mean"]) / q_stats["std"]
        
        # Update context: append q_norm if not present, replace if it is
        if len(context) == 3:
            new_context = np.append(context, q_norm)
        else:
            new_context = np.copy(context)
            new_context[3] = q_norm
        
        # Update unnormalized context: append q if not present
        if len(unnorm_context) == 3:
            new_unnorm_context = np.append(unnorm_context, q)
        else:
            new_unnorm_context = unnorm_context  # Already has q
        
        # Save updated files
        np.save(context_file, new_context)
        np.save(unnorm_file, new_unnorm_context)

def main():
    # Get all context files
    context_files, unnorm_context_files = get_context_files(BASE_PATH, DATA_TYPES)
    if not context_files:
        print("Error: No context files found. Check BASE_PATH and DATA_TYPES.")
        return
    
    print(f"Found {len(context_files)} context files to process.")

    # Compute q stats
    q_stats, q_values = compute_q_stats(context_files, unnorm_context_files)
    
    # Print statistics
    print(f"Dynamic Pressure (q) Statistics:")
    print(f"  Min: {q_stats['min']:.2f} Pa")
    print(f"  Max: {q_stats['max']:.2f} Pa")
    print(f"  Mean: {q_stats['mean']:.2f} Pa")
    print(f"  Std: {q_stats['std']:.2f} Pa")

    # Save q stats
    os.makedirs(os.path.dirname(Q_STATS_FILE), exist_ok=True)
    with open(Q_STATS_FILE, "w") as f:
        json.dump(q_stats, f)
    print(f"Saved q statistics to {Q_STATS_FILE}")

    # Update context files with normalized q
    update_context_files(context_files, unnorm_context_files, q_values, q_stats)
    print("Updated all context files with normalized q values.")

if __name__ == "__main__":
    main()