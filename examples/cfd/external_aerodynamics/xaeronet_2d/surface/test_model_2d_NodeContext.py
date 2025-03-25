import os
import json
import torch
import dgl
import pyvista as pv
import numpy as np
from torch.amp import autocast
from modulus.models.meshgraphnet import MeshGraphNet
from dataloader_NodeContext import create_dataloader
from utils import find_bin_files
from compute_CLCD_Cm import compute_cl_cd
from plot_CLCD import plot_cl_cd_comparison, plot_cl_cd_colorVars
from hydra.utils import to_absolute_path
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

def compute_error_metrics(cl_cd_results_list):
    """
    Compute various error metrics for CL, CD, and Cmy predictions.

    Args:
        cl_cd_results_list (list): List of dictionaries with CL_pred, CL_true, CD_pred, CD_true,
                                   Cmy_pred, and Cmy_true.

    Returns:
        dict: Various error metrics for CL, CD, and Cmy.
    """
    # Extract predicted and true values for CL, CD, and Cmy
    cl_pred = np.array([r["CL_pred"] for r in cl_cd_results_list])
    cl_true = np.array([r["CL_true"] for r in cl_cd_results_list])
    cd_pred = np.array([r["CD_pred"] for r in cl_cd_results_list])
    cd_true = np.array([r["CD_true"] for r in cl_cd_results_list])
    cmy_pred = np.array([r["Cmy_pred"] for r in cl_cd_results_list])
    cmy_true = np.array([r["Cmy_true"] for r in cl_cd_results_list])

    # Define a small epsilon to prevent division by zero in relative error calculations
    epsilon = 1e-6

    # Compute relative errors with epsilon for numerical stability
    cl_rel_errors = np.abs(cl_pred - cl_true) / (np.abs(cl_true) + epsilon)
    cd_rel_errors = np.abs(cd_pred - cd_true) / (np.abs(cd_true) + epsilon)
    cmy_rel_errors = np.abs(cmy_pred - cmy_true) / (np.abs(cmy_true) + epsilon)

    # Define all error metrics for CL, CD, and Cmy
    metrics = {
        # CL metrics
        "max_cl_error": np.max(cl_rel_errors),
        "mean_cl_error": np.mean(cl_rel_errors),
        "rmsre_cl": np.sqrt(np.mean(cl_rel_errors**2)),
        "mae_cl": np.mean(np.abs(cl_pred - cl_true)),
        "r2_cl": 1 - np.sum((cl_true - cl_pred)**2) / np.sum((cl_true - np.mean(cl_true))**2),
        "median_cl_error": np.median(cl_rel_errors),
        "percent_cl_within": np.mean(cl_rel_errors < 0.1) * 100,

        # CD metrics
        "max_cd_error": np.max(cd_rel_errors),
        "mean_cd_error": np.mean(cd_rel_errors),
        "rmsre_cd": np.sqrt(np.mean(cd_rel_errors**2)),
        "mae_cd": np.mean(np.abs(cd_pred - cd_true)),
        "r2_cd": 1 - np.sum((cd_true - cd_pred)**2) / np.sum((cd_true - np.mean(cd_true))**2),
        "median_cd_error": np.median(cd_rel_errors),
        "percent_cd_within": np.mean(cd_rel_errors < 0.1) * 100,

        # Cmy metrics
        "max_cmy_error": np.max(cmy_rel_errors),
        "mean_cmy_error": np.mean(cmy_rel_errors),
        "rmsre_cmy": np.sqrt(np.mean(cmy_rel_errors**2)),
        "mae_cmy": np.mean(np.abs(cmy_pred - cmy_true)),
        "r2_cmy": 1 - np.sum((cmy_true - cmy_pred)**2) / np.sum((cmy_true - np.mean(cmy_true))**2),
        "median_cmy_error": np.median(cmy_rel_errors),
        "percent_cmy_within": np.mean(cmy_rel_errors < 0.1) * 100,
    }
    return metrics

def plot_test_cases(vtp_files, output_dir):
    """
    Generate three plots for each test case, each with three subplots for pressure and shear stress components.

    Args:
        vtp_files (list): List of paths to .vtp files.
        output_dir (str): Directory to save the plots.
    """
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for vtp_file in vtp_files:
        # Load point cloud
        point_cloud = pv.read(vtp_file)
        coordinates = point_cloud.points[:, :2]  # (x, y) coordinates
        test_id = os.path.basename(vtp_file).split('_')[-1].split('.')[0]
        aoa = point_cloud["AOA"][0]
        mach = point_cloud["Mach"][0]
        rel = point_cloud["ReL"][0]

        # Data for plotting
        data_sets = [
            {
                "name": "Pressure",
                "true": point_cloud["pressure_true"],
                "pred": point_cloud["pressure_pred"],
                "diff": point_cloud["pressure_diff"],  # Relative error
            },
            {
                "name": "Shear_Stress_X",
                "true": point_cloud["shear_stress_true"][:, 0],
                "pred": point_cloud["shear_stress_pred"][:, 0],
                "diff": point_cloud["shear_stress_diff"][:, 0],
            },
            {
                "name": "Shear_Stress_Y",
                "true": point_cloud["shear_stress_true"][:, 1],
                "pred": point_cloud["shear_stress_pred"][:, 1],
                "diff": point_cloud["shear_stress_diff"][:, 1],
            },
        ]

        # Generate one figure per variable
        for data in data_sets:
            fig, axes = plt.subplots(3, 1)
            vmin = np.min([data["true"], data["pred"]])
            vmax = np.max([data["true"], data["pred"]])
            # vmin_diff = np.min(data["diff"])
            vmax_diff = np.max(data["diff"])

            # True subplot
            sc0 = axes[0].scatter(coordinates[:, 0], coordinates[:, 1], c=data["true"], s=1, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[0].set_title("True")
            fig.colorbar(sc0, ax=axes[0])

            # Predicted subplot
            sc1 = axes[1].scatter(coordinates[:, 0], coordinates[:, 1], c=data["pred"], s=1, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[1].set_title("Predicted")
            fig.colorbar(sc1, ax=axes[1])

            # Difference subplot (relative error)
            sc2 = axes[2].scatter(coordinates[:, 0], coordinates[:, 1], c=data["diff"], s=1, cmap='viridis', vmin=0, vmax=vmax_diff)
            axes[2].set_title("Diff")
            fig.colorbar(sc2, ax=axes[2])

            # Labels and title
            for ax in axes:
                ax.set_xlabel('x')
                ax.set_ylabel('y')
            fig.suptitle(f"{data['name']} - Test Case {test_id}\nAOA: {aoa:.2f}, Mach: {mach:.2f}, ReL: {rel:.2e}")

            # Save plot
            plt.tight_layout()
            plot_file = os.path.join(plot_dir, f"plot_{data['name']}_{test_id}.png")
            plt.savefig(plot_file)
            plt.close(fig)
            # print(f"Saved plot to {plot_file}")

def evaluate_test_data(cfg: DictConfig, model_checkpoint: str, output_dir: str = "test_point_clouds"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cl_cd_results_list = []
    vtp_files = []
    os.makedirs(output_dir, exist_ok=True)

    with open(to_absolute_path(cfg.stats_file), "r") as f:
        stats = json.load(f)
    mean = stats["mean"]
    std = stats["std_dev"]

    normalize = {
        "coordinates": True, "normals": True, "area": True, "pressure": True,
        "shear_stress": True, "x": True, "Mach": True, "ReL": True, "AOA": True
    }

    test_dataset = find_bin_files(to_absolute_path(cfg.test_partitions_path))
    test_dataloader = create_dataloader(test_dataset, mean, std, batch_size=1, prefetch_factor=None, use_ddp=False, num_workers=4)
    test_graphs = [graph_partitions for graph_partitions, _ in test_dataloader]
    test_ids = [id[0] for _, id in test_dataloader]

    model = MeshGraphNet(
        input_dim_nodes=19, input_dim_edges=3, output_dim=3, processor_size=cfg.num_message_passing_layers,
        hidden_dim_processor=cfg.processor_hidden_dim, aggregation="sum", hidden_dim_node_encoder=cfg.hidden_dim,
        hidden_dim_edge_encoder=cfg.hidden_dim, hidden_dim_node_decoder=cfg.hidden_dim, mlp_activation_fn=cfg.activation,
        do_concat_trick=cfg.use_concat_trick, num_processor_checkpoint_segments=cfg.checkpoint_segments,
    ).to(device)
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    amp_dtype = torch.bfloat16
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, subgraphs in enumerate(test_graphs):
        num_nodes = sum(subgraph.num_nodes() for subgraph in subgraphs)
        sum_unique_nodes = sum(len(subgraphs[j].ndata[dgl.NID][subgraphs[j].ndata["inner_node"].bool()]) for j in range(cfg.num_partitions))

        pressure_pred = torch.zeros((sum_unique_nodes, 1), dtype=torch.float32, device=device)
        shear_stress_pred = torch.zeros((sum_unique_nodes, 2), dtype=torch.float32, device=device)
        pressure_true = torch.zeros((sum_unique_nodes, 1), dtype=torch.float32, device=device)
        shear_stress_true = torch.zeros((sum_unique_nodes, 2), dtype=torch.float32, device=device)
        coordinates = torch.zeros((sum_unique_nodes, 2), dtype=torch.float32, device=device)
        normals = torch.zeros((sum_unique_nodes, 2), dtype=torch.float32, device=device)
        area = torch.zeros((sum_unique_nodes, 1), dtype=torch.float32, device=device)
        Mach = torch.zeros((sum_unique_nodes,), dtype=torch.float32, device=device)
        ReL = torch.zeros((sum_unique_nodes,), dtype=torch.float32, device=device)
        AOA = torch.zeros((sum_unique_nodes,), dtype=torch.float32, device=device)

        for j in range(cfg.num_partitions):
            part = subgraphs[j].to(device)
            ndata = torch.cat(
                (
                    part.ndata["coordinates"], part.ndata["normals"],
                    torch.sin(2 * np.pi * part.ndata["coordinates"]), torch.cos(2 * np.pi * part.ndata["coordinates"]),
                    torch.sin(4 * np.pi * part.ndata["coordinates"]), torch.cos(4 * np.pi * part.ndata["coordinates"]),
                    torch.sin(8 * np.pi * part.ndata["coordinates"]), torch.cos(8 * np.pi * part.ndata["coordinates"]),
                    part.ndata["Mach"].view(-1, 1), part.ndata["ReL"].view(-1, 1), part.ndata["AOA"].view(-1, 1),
                ),
                dim=1,
            )

            with torch.no_grad():
                with autocast(amp_device, enabled=True, dtype=amp_dtype):
                    pred = model(ndata, part.edata["x"], part)
                    pred_filtered = pred[part.ndata["inner_node"].bool()]
                    target = torch.cat((part.ndata["pressure"], part.ndata["shear_stress"]), dim=1)
                    target_filtered = target[part.ndata["inner_node"].bool()]

                    original_nodes = part.ndata[dgl.NID]
                    inner_original_nodes = original_nodes[part.ndata["inner_node"].bool()]
                    pressure_pred[inner_original_nodes] = pred_filtered[:, 0:1].clone().to(torch.float32)
                    shear_stress_pred[inner_original_nodes] = pred_filtered[:, 1:].clone().to(torch.float32)
                    pressure_true[inner_original_nodes] = target_filtered[:, 0:1].clone().to(torch.float32)
                    shear_stress_true[inner_original_nodes] = target_filtered[:, 1:].clone().to(torch.float32)
                    coordinates[original_nodes] = part.ndata["coordinates"].clone().to(torch.float32)
                    normals[original_nodes] = part.ndata["normals"].clone().to(torch.float32)
                    area[original_nodes] = part.ndata["area"].clone().to(torch.float32)
                    Mach[original_nodes] = part.ndata["Mach"].clone().to(torch.float32)
                    ReL[original_nodes] = part.ndata["ReL"].clone().to(torch.float32)
                    AOA[original_nodes] = part.ndata["AOA"].clone().to(torch.float32)

        pressure_pred_denorm = (pressure_pred.cpu() * torch.tensor(std["pressure"])) + torch.tensor(mean["pressure"]) if normalize["pressure"] else pressure_pred.cpu()
        pressure_true_denorm = (pressure_true.cpu() * torch.tensor(std["pressure"])) + torch.tensor(mean["pressure"]) if normalize["pressure"] else pressure_true.cpu()
        shear_stress_pred_denorm = (shear_stress_pred.cpu() * torch.tensor(std["shear_stress"])) + torch.tensor(mean["shear_stress"]) if normalize["shear_stress"] else shear_stress_pred.cpu()
        shear_stress_true_denorm = (shear_stress_true.cpu() * torch.tensor(std["shear_stress"])) + torch.tensor(mean["shear_stress"]) if normalize["shear_stress"] else shear_stress_true.cpu()
        coordinates_denorm = (coordinates.cpu() * torch.tensor(std["coordinates"])) + torch.tensor(mean["coordinates"]) if normalize["coordinates"] else coordinates.cpu()
        normals_denorm = (normals.cpu() * torch.tensor(std["normals"])) + torch.tensor(mean["normals"]) if normalize["normals"] else normals.cpu()
        area_denorm = (area.cpu() * torch.tensor(std["area"])) + torch.tensor(mean["area"]) if normalize["area"] else area_denorm.cpu()
        Mach_denorm = (Mach.cpu() * torch.tensor(std["Mach"])) + torch.tensor(mean["Mach"]) if normalize["Mach"] else Mach.cpu()
        ReL_denorm = (ReL.cpu() * torch.tensor(std["ReL"])) + torch.tensor(mean["ReL"]) if normalize["ReL"] else ReL.cpu()
        AOA_denorm = (AOA.cpu() * torch.tensor(std["AOA"])) + torch.tensor(mean["AOA"]) if normalize["AOA"] else AOA.cpu()

        coordinates_denorm = np.column_stack((coordinates_denorm.numpy(), np.zeros((coordinates_denorm.shape[0], 1))))
        normals_denorm = np.column_stack((normals_denorm.numpy(), np.zeros((normals_denorm.shape[0], 1))))
        shear_stress_pred_denorm = np.column_stack((shear_stress_pred_denorm.numpy(), np.zeros((shear_stress_pred_denorm.shape[0], 1))))
        shear_stress_true_denorm = np.column_stack((shear_stress_true_denorm.numpy(), np.zeros((shear_stress_true_denorm.shape[0], 1))))

        point_cloud = pv.PolyData(coordinates_denorm)
        point_cloud["coordinates"] = coordinates_denorm
        point_cloud["normals"] = normals_denorm
        point_cloud["area"] = area_denorm.numpy()
        point_cloud["Mach"] = Mach_denorm.numpy()
        point_cloud["AOA"] = AOA_denorm.numpy()
        point_cloud["ReL"] = ReL_denorm.numpy()
        point_cloud["pressure_pred"] = pressure_pred_denorm.numpy()
        point_cloud["shear_stress_pred"] = shear_stress_pred_denorm
        point_cloud["pressure_true"] = pressure_true_denorm.numpy()
        point_cloud["shear_stress_true"] = shear_stress_true_denorm
        epsilon = 1e-6
        point_cloud["shear_stress_err"] = np.abs((shear_stress_true_denorm - shear_stress_pred_denorm) / (shear_stress_true_denorm + epsilon))
        point_cloud["pressure_err"] = np.abs((pressure_true_denorm.numpy() - pressure_pred_denorm.numpy()) / (pressure_true_denorm.numpy() + epsilon))
        point_cloud["shear_stress_diff"] = np.abs((shear_stress_true_denorm - shear_stress_pred_denorm))
        point_cloud["pressure_diff"] = np.abs((pressure_true_denorm.numpy() - pressure_pred_denorm.numpy()))

        vtp_file = os.path.join(output_dir, f"point_cloud_output_{test_ids[i]}.vtp")
        point_cloud.save(vtp_file)
        vtp_files.append(vtp_file)
        cl_cd_results = compute_cl_cd(vtp_file)
        cl_cd_results_list.append(cl_cd_results)

    print(f"Test evaluation complete. Point clouds saved to {output_dir}")
    return cl_cd_results_list, vtp_files

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    output_dir = "/workspace/outputs/XAeroNetS/test_point_clouds"
    cl_cd_results_list, vtp_files = evaluate_test_data(
        cfg=cfg,
        model_checkpoint="/workspace/outputs/XAeroNetS/final_model_checkpoint.pth",
        output_dir=output_dir
    )

    plot_cl_cd_comparison(cl_cd_results_list)
    plot_cl_cd_colorVars(cl_cd_results_list)
    plot_test_cases(vtp_files, output_dir)
    error_metrics = compute_error_metrics(cl_cd_results_list)

    print("Error Metrics:")
    print(f"Max Relative Error (CL): {error_metrics['max_cl_error']:.4f}")
    print(f"Max Relative Error (CD): {error_metrics['max_cd_error']:.4f}")
    print(f"Max Relative Error (Cmy): {error_metrics['max_cmy_error']:.4f}")
    print(f"Mean Relative Error (CL): {error_metrics['mean_cl_error']:.4f}")
    print(f"Mean Relative Error (CD): {error_metrics['mean_cd_error']:.4f}")
    print(f"Mean Relative Error (Cmy): {error_metrics['mean_cmy_error']:.4f}")
    print(f"Root Mean Squared Relative Error (CL): {error_metrics['rmsre_cl']:.4f}")
    print(f"Root Mean Squared Relative Error (CD): {error_metrics['rmsre_cd']:.4f}")
    print(f"Root Mean Squared Relative Error (Cmy): {error_metrics['rmsre_cmy']:.4f}")
    print(f"Mean Absolute Error (CL): {error_metrics['mae_cl']:.4f}")  # Corrected label from "Max" to "Mean"
    print(f"Mean Absolute Error (CD): {error_metrics['mae_cd']:.4f}")  # Corrected label from "Max" to "Mean"
    print(f"Mean Absolute Error (Cmy): {error_metrics['mae_cmy']:.4f}")
    print(f"Coefficient of Determination - R² (CL): {error_metrics['r2_cl']:.4f}")
    print(f"Coefficient of Determination - R² (CD): {error_metrics['r2_cd']:.4f}")
    print(f"Coefficient of Determination - R² (Cmy): {error_metrics['r2_cmy']:.4f}")
    print(f"Median Relative Error (CL): {error_metrics['median_cl_error']:.4f}")
    print(f"Median Relative Error (CD): {error_metrics['median_cd_error']:.4f}")
    print(f"Median Relative Error (Cmy): {error_metrics['median_cmy_error']:.4f}")
    print(f"% CL within 10% error: {error_metrics['percent_cl_within']:.2f}%")
    print(f"% CD within 10% error: {error_metrics['percent_cd_within']:.2f}%")
    print(f"% Cmy within 10% error: {error_metrics['percent_cmy_within']:.2f}%")

    # Specify the output file
    output_file = "/workspace/outputs/XAeroNetS/error_metrics_results.txt"

    # Open the file in write mode and write the results
    with open(output_file, 'w') as f:
        f.write("Error Metrics:\n")
        f.write(f"Max Relative Error (CL): {error_metrics['max_cl_error']:.4f}\n")
        f.write(f"Max Relative Error (CD): {error_metrics['max_cd_error']:.4f}\n")
        f.write(f"Max Relative Error (Cmy): {error_metrics['max_cmy_error']:.4f}\n")
        f.write(f"Mean Relative Error (CL): {error_metrics['mean_cl_error']:.4f}\n")
        f.write(f"Mean Relative Error (CD): {error_metrics['mean_cd_error']:.4f}\n")
        f.write(f"Mean Relative Error (Cmy): {error_metrics['mean_cmy_error']:.4f}\n")
        f.write(f"Root Mean Squared Relative Error (CL): {error_metrics['rmsre_cl']:.4f}\n")
        f.write(f"Root Mean Squared Relative Error (CD): {error_metrics['rmsre_cd']:.4f}\n")
        f.write(f"Root Mean Squared Relative Error (Cmy): {error_metrics['rmsre_cmy']:.4f}\n")
        f.write(f"Mean Absolute Error (CL): {error_metrics['mae_cl']:.4f}\n")  # Corrected label
        f.write(f"Mean Absolute Error (CD): {error_metrics['mae_cd']:.4f}\n")  # Corrected label
        f.write(f"Mean Absolute Error (Cmy): {error_metrics['mae_cmy']:.4f}\n")
        f.write(f"Coefficient of Determination - R² (CL): {error_metrics['r2_cl']:.4f}\n")
        f.write(f"Coefficient of Determination - R² (CD): {error_metrics['r2_cd']:.4f}\n")
        f.write(f"Coefficient of Determination - R² (Cmy): {error_metrics['r2_cmy']:.4f}\n")
        f.write(f"Median Relative Error (CL): {error_metrics['median_cl_error']:.4f}\n")
        f.write(f"Median Relative Error (CD): {error_metrics['median_cd_error']:.4f}\n")
        f.write(f"Median Relative Error (Cmy): {error_metrics['median_cmy_error']:.4f}\n")
        f.write(f"% CL within 10% error: {error_metrics['percent_cl_within']:.2f}%\n")
        f.write(f"% CD within 10% error: {error_metrics['percent_cd_within']:.2f}%\n")
        f.write(f"% Cmy within 10% error: {error_metrics['percent_cmy_within']:.2f}%\n")

    print(f"Error metrics results have been written to {output_file}")
if __name__ == "__main__":
    main()