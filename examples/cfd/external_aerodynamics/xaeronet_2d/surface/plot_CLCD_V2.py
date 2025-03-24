import matplotlib.pyplot as plt
import numpy as np
from compute_CLCD import compute_cl_cd

def plot_cl_cd_comparison(cl_cd_list):
    """
    Plot a 5D comparison of CL and CD with AOA (color), Mach (size), and ReL (facets).

    Args:
        cl_cd_list (list): List of dictionaries containing CL_pred, CL_true, CD_pred, CD_true, AOA, ReL, Mach.
    """
    # Extract data
    cl_pred = np.array([r["CL_pred"] for r in cl_cd_list])
    cl_true = np.array([r["CL_true"] for r in cl_cd_list])
    cd_pred = np.array([r["CD_pred"] for r in cl_cd_list])
    cd_true = np.array([r["CD_true"] for r in cl_cd_list])
    aoa = np.array([r["AOA"] for r in cl_cd_list])
    mach = np.array([r["Mach"] for r in cl_cd_list])
    rel = np.array([r["ReL"] for r in cl_cd_list])

    # Bin ReL into discrete categories (e.g., quartiles) if continuous
    rel_bins = np.percentile(rel, [0, 25, 50, 75, 100])  # Adjust bins as needed
    rel_digitized = np.digitize(rel, bins=rel_bins, right=True)
    unique_rel = np.unique(rel_digitized)

    # Normalize Mach for point sizes (between 50 and 500, for example)
    mach_normalized = 50 + 450 * (mach - mach.min()) / (mach.max() - mach.min() + 1e-6)

    # Create figure with subplots for each ReL bin
    num_cols = len(unique_rel)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), sharey='row')

    # Handle case of single ReL value
    if num_cols == 1:
        axes = np.array([axes]).T  # Ensure axes is 2D

    # Plot for each ReL bin
    for i, rel_bin in enumerate(unique_rel):
        mask = rel_digitized == rel_bin
        rel_value = rel[mask][0]  # Representative value for label

        # CL Plot (top row)
        ax_cl = axes[0, i]
        scatter_cl = ax_cl.scatter(
            cl_true[mask], cl_pred[mask], c=aoa[mask], s=mach_normalized[mask],
            cmap='viridis', label='Predicted vs True'
        )
        ax_cl.plot([min(cl_true), max(cl_true)], [min(cl_true), max(cl_true)], 'r--', label='Ideal')
        ax_cl.set_xlabel('True CL')
        ax_cl.set_ylabel('Predicted CL')
        ax_cl.set_title(f'Lift (ReL ≈ {rel_value:.2e})')
        ax_cl.legend()
        plt.colorbar(scatter_cl, ax=ax_cl, label='AOA')

        # CD Plot (bottom row)
        ax_cd = axes[1, i]
        scatter_cd = ax_cd.scatter(
            cd_true[mask], cd_pred[mask], c=aoa[mask], s=mach_normalized[mask],
            cmap='viridis', label='Predicted vs True'
        )
        ax_cd.plot([min(cd_true), max(cd_true)], [min(cd_true), max(cd_true)], 'r--', label='Ideal')
        ax_cd.set_xlabel('True CD')
        ax_cd.set_ylabel('Predicted CD')
        ax_cd.set_title(f'Drag (ReL ≈ {rel_value:.2e})')
        ax_cd.legend()
        plt.colorbar(scatter_cd, ax=ax_cd, label='AOA')

    plt.tight_layout()

    # Save with a descriptive filename
    save_path = "test_point_clouds/CLCD_scatter_5D_AOA_Mach_ReL.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.show()

# Example usage
# if __name__ == "__main__":
#     cl_cd_list = [compute_cl_cd("test_point_clouds/point_cloud_test_id.vtp")]
#     plot_cl_cd_comparison(cl_cd_list)