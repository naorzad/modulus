# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import pyvista as pv
import numpy as np

def compute_cl_cd(vtp_file: str):
    """
    Compute CL and CD from a .vtp file containing pressure, shear stress, normals, area, and AOA.
    
    Args:
        vtp_file (str): Path to the .vtp file.
        rho (float): Air density (kg/m^3), default 1.225 for standard air.
        V (float): Freestream velocity (m/s), default 1.0 (adjust based on Mach/ReL).
        S_ref (float): Reference area (m^2), default 1.0 (typically chord length in 2D).
    
    Returns:
        dict: Predicted and true CL, CD values.
    """
    # Load point cloud
    point_cloud = pv.read(vtp_file)

    # Extract data
    pressure_pred = point_cloud["pressure_pred"].flatten()
    pressure_true = point_cloud["pressure_true"].flatten()  # If used elsewhere
    shear_stress_pred = point_cloud["shear_stress_pred"][:, :2]  # 2D
    shear_stress_true = point_cloud["shear_stress_true"][:, :2]  # 2D
    area = point_cloud["area"].flatten()
    normals = point_cloud["normals"][:, :2]  # (320, 2) for 2D normals
    AOA = np.mean(point_cloud["AOA"])  # Assume uniform AOA across points

    # Pressure forces (F_p = -pressure * area * normal)
    C_f_pred = -pressure_pred[:, np.newaxis] * area[:, np.newaxis] * normals
    C_f_true = -pressure_true[:, np.newaxis] * area[:, np.newaxis] * normals

    # Shear forces (F_s = shear_stress * area)
    C_s_pred = shear_stress_pred * area[:, np.newaxis]
    C_s_true = shear_stress_true * area[:, np.newaxis]

    # Total forces
    C_total_pred = C_f_pred + C_s_pred
    C_total_true = C_f_true + C_s_true

    # Sum forces over all elements
    C_total_pred_sum = np.sum(C_total_pred, axis=0)
    C_total_true_sum = np.sum(C_total_true, axis=0)

    # Rotate forces into lift and drag directions based on AOA (in radians)
    AOA_rad = np.deg2rad(AOA)
    cos_a = np.cos(AOA_rad)
    sin_a = np.sin(AOA_rad)
    
    # Lift direction: perpendicular to freestream (y-dir rotated by AOA)
    # Drag direction: parallel to freestream (x-dir rotated by AOA)
    CL_pred = C_total_pred_sum[1] * cos_a - C_total_pred_sum[0] * sin_a
    CD_pred = C_total_pred_sum[0] * cos_a + C_total_pred_sum[1] * sin_a
    CL_true = C_total_true_sum[1] * cos_a - C_total_true_sum[0] * sin_a
    CD_true = C_total_true_sum[0] * cos_a + C_total_true_sum[1] * sin_a

    return {
        "CL_pred": CL_pred,
        "CD_pred": CD_pred,
        "CL_true": CL_true,
        "CD_true": CD_true,
        "AOA": point_cloud["AOA"][0],
        "ReL": point_cloud["ReL"][0],
        "Mach": point_cloud["Mach"][0],
    }

if __name__ == "__main__":
    # Example usage
    result = compute_cl_cd("test_point_clouds/point_cloud_test_id.vtp")
    print(result)