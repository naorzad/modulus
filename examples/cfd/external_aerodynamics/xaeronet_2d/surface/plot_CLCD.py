import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

def plot_cl_cd_comparison(cl_cd_list,output_dir):
    """
    Plot a 5D comparison of CL, CD, and Cmy with AOA (color), Mach (size), and ReL (facets).

    Args:
        cl_cd_list (list): List of dictionaries containing CL_pred, CL_true, CD_pred, CD_true,
                           Cmy_pred, Cmy_true, AOA, ReL, Mach.
    """
    # Extract data
    cl_pred = np.array([r["CL_pred"] for r in cl_cd_list])
    cl_true = np.array([r["CL_true"] for r in cl_cd_list])
    cd_pred = np.array([r["CD_pred"] for r in cl_cd_list])
    cd_true = np.array([r["CD_true"] for r in cl_cd_list])
    cmy_pred = np.array([r["Cmy_pred"] for r in cl_cd_list])
    cmy_true = np.array([r["Cmy_true"] for r in cl_cd_list])
    aoa = np.array([r["AOA"] for r in cl_cd_list])
    mach = np.array([r["Mach"] for r in cl_cd_list])
    rel = np.array([r["ReL"] for r in cl_cd_list])

    # Bin ReL into discrete categories (e.g., quartiles)
    rel_bins = np.percentile(rel, [0, 25, 50, 75, 100])  # Adjust bins as needed
    rel_digitized = np.digitize(rel, bins=rel_bins, right=True)
    unique_rel = np.unique(rel_digitized)

    # Normalize Mach for point sizes (between 50 and 500)
    mach_normalized = 50 + 450 * (mach - mach.min()) / (mach.max() - mach.min() + 1e-6)

    # Create figure with subplots for each ReL bin, now 3 rows for CL, CD, Cmy
    num_cols = len(unique_rel)
    fig, axes = plt.subplots(3, num_cols, figsize=(5 * num_cols, 15), sharey='row')  # Changed to 3 rows

    # Handle case of single ReL value
    if num_cols == 1:
        axes = np.array([axes]).T  # Ensure axes is 2D

    # Plot for each ReL bin
    for i, rel_bin in enumerate(unique_rel):
        mask = rel_digitized == rel_bin
        rel_value = rel[mask][0]  # Representative value for label

        # CL Plot (first row)
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

        # CD Plot (second row)
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

        # Cmy Plot (third row)
        ax_cmy = axes[2, i]
        scatter_cmy = ax_cmy.scatter(
            cmy_true[mask], cmy_pred[mask], c=aoa[mask], s=mach_normalized[mask],
            cmap='viridis', label='Predicted vs True'
        )
        ax_cmy.plot([min(cmy_true), max(cmy_true)], [min(cmy_true), max(cmy_true)], 'r--', label='Ideal')
        ax_cmy.set_xlabel('True Cmy')
        ax_cmy.set_ylabel('Predicted Cmy')
        ax_cmy.set_title(f'Pitching Moment (ReL ≈ {rel_value:.2e})')
        ax_cmy.legend()
        plt.colorbar(scatter_cmy, ax=ax_cmy, label='AOA')

    plt.tight_layout()

    # Save with a descriptive filename
    save_path = os.path.join(output_dir,"CLCD_Cmy_scatter_5D_AOA_Mach_ReL.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_cl_cd_colorVars(cl_cd_list,output_dir):
    """
    Plot CL, CD, and Cmy comparison with colorbars for all variables: AOA, ReL, and Mach.

    Args:
        cl_cd_list (list): List of dictionaries containing CL_pred, CL_true, CD_pred, CD_true,
                           Cmy_pred, Cmy_true, AOA, ReL, Mach.
    """
    # Extract data
    cl_pred = [r["CL_pred"] for r in cl_cd_list]
    cl_true = [r["CL_true"] for r in cl_cd_list]
    cd_pred = [r["CD_pred"] for r in cl_cd_list]
    cd_true = [r["CD_true"] for r in cl_cd_list]
    cmy_pred = [r["Cmy_pred"] for r in cl_cd_list]
    cmy_true = [r["Cmy_true"] for r in cl_cd_list]
    
    # Define all variables to plot
    variables = ['AOA', 'ReL', 'Mach']
    color_values = {
        'AOA': [r["AOA"] for r in cl_cd_list],
        'ReL': [r["ReL"] for r in cl_cd_list],
        'Mach': [r["Mach"] for r in cl_cd_list]
    }

    # Loop over each variable
    for var in variables:
        # Create a new figure for each variable, now with 3 subplots
        plt.figure(figsize=(18, 5))  # Increased width for 3 subplots
        
        # CL Plot
        plt.subplot(1, 3, 1)  # Changed to 1 row, 3 columns
        scatter = plt.scatter(cl_true, cl_pred, c=color_values[var], cmap='viridis', label='Predicted vs True')
        plt.colorbar(scatter, label=var)
        plt.plot([min(cl_true), max(cl_true)], [min(cl_true), max(cl_true)], 'r--', label='Ideal')
        plt.xlabel('True CL')
        plt.ylabel('Predicted CL')
        plt.title('Lift Coefficient Comparison')
        plt.legend()

        # CD Plot
        plt.subplot(1, 3, 2)
        scatter = plt.scatter(cd_true, cd_pred, c=color_values[var], cmap='viridis', label='Predicted vs True')
        plt.colorbar(scatter, label=var)
        plt.plot([min(cd_true), max(cd_true)], [min(cd_true), max(cd_true)], 'r--', label='Ideal')
        plt.xlabel('True CD')
        plt.ylabel('Predicted CD')
        plt.title('Drag Coefficient Comparison')
        plt.legend()

        # Cmy Plot
        plt.subplot(1, 3, 3)
        scatter = plt.scatter(cmy_true, cmy_pred, c=color_values[var], cmap='viridis', label='Predicted vs True')
        plt.colorbar(scatter, label=var)
        plt.plot([min(cmy_true), max(cmy_true)], [min(cmy_true), max(cmy_true)], 'r--', label='Ideal')
        plt.xlabel('True Cmy')
        plt.ylabel('Predicted Cmy')
        plt.title('Pitching Moment Coefficient Comparison')
        plt.legend()

        plt.tight_layout()
        
        # Save the figure with the variable name in the filename
        save_path = os.path.join(output_dir,f"CLCD_Cmy_scatter_{var}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot for {var} saved to {save_path}")
        
        plt.close()

def plot_coefficients_vs_aoa(cl_cd_list, output_dir):
    """
    Plot CL, CD, and CM versus AOA for each unique ReL value, with Mach indicated by color.

    Args:
        cl_cd_list (list): List of dictionaries containing CL_true, CL_pred, CD_true, CD_pred,
                           Cmy_true, Cmy_pred, AOA, ReL, Mach.
        output_dir (str): Directory to save the plots.
    """
    # Extract data from the input list
    aoa = np.array([r["AOA"] for r in cl_cd_list])
    mach = np.array([r["Mach"] for r in cl_cd_list])
    rel = np.array([r["ReL"] for r in cl_cd_list])

    # Get unique ReL values
    unique_rel = np.unique(rel)

    # Normalize Mach values for consistent color mapping across all figures
    mach_min, mach_max = mach.min(), mach.max()
    norm = Normalize(mach_min, mach_max)
    cmap = plt.get_cmap('viridis')

    # Define coefficients to plot
    coefficients = [
        ('CL', 'CL_true', 'CL_pred'),
        ('CD', 'CD_true', 'CD_pred'),
        ('CM', 'Cmy_true', 'Cmy_pred')
    ]

    # Loop over each unique ReL value
    for rel_val in unique_rel:
        # Create a new figure with 3 subplots (one for each coefficient)
        fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

        # Filter data for this ReL value
        mask = rel == rel_val
        subset_aoa = aoa[mask]
        subset_mach = mach[mask]

        # Plot each coefficient
        for row, (coef_name, true_key, pred_key) in enumerate(coefficients):
            ax = axes[row]
            subset_true = np.array([r[true_key] for r in cl_cd_list])[mask]
            subset_pred = np.array([r[pred_key] for r in cl_cd_list])[mask]

            # Plot for each unique Mach value in this ReL
            unique_mach = np.unique(subset_mach)
            for m in unique_mach:
                mach_mask = subset_mach == m
                # Sort by AOA to ensure lines connect points in order
                sorted_indices = np.argsort(subset_aoa[mach_mask])
                aoa_sorted = subset_aoa[mach_mask][sorted_indices]
                true_sorted = subset_true[mach_mask][sorted_indices]
                pred_sorted = subset_pred[mach_mask][sorted_indices]

                # Assign color based on Mach
                color = cmap(norm(m))
                # Plot true values as lines, predicted as markers
                if len(aoa_sorted) > 1:
                    ax.plot(aoa_sorted, true_sorted, color=color)
                else:
                    ax.scatter(aoa_sorted, true_sorted, color=color, marker='x')
                ax.scatter(aoa_sorted, pred_sorted, color=color, marker='o', alpha=0.7)

            # Customize subplot labels and appearance
            ax.set_ylabel(coef_name)
            if row == 2:
                ax.set_xlabel('AOA (degrees)')
            ax.grid(True)

        # Add a vertical colorbar for Mach
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Mach', orientation='vertical', fraction=0.02, pad=0.05)

        # Add a suptitle with the ReL value
        fig.suptitle(f'ReL = {rel_val:.2e}', fontsize=16)

        # Add a text annotation to explain lines vs. markers
        fig.text(0.5, 0.92, "Lines: true values, Markers: predicted values", ha='center', va='center')

        # Manually adjust layout to avoid overlapping
        plt.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1, hspace=0.2)

        # Save the plot with a descriptive filename
        save_path = os.path.join(output_dir, f"coefficients_vs_AOA_ReL_{rel_val:.2e}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

        # Close the figure to free memory
        plt.close(fig)
    
# Example usage
if __name__ == "__main__":
    # Assuming cl_cd_list is populated from your test script with Cmy data
    cl_cd_list = [compute_cl_cd_cmy("test_point_clouds/point_cloud_test_id.vtp", "path/to/stats.json")]
    plot_cl_cd_comparison(cl_cd_list)
    plot_cl_cd_colorVars(cl_cd_list)