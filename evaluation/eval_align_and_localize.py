import numpy as np
import open3d as o3d
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]   # VisionPipeline/

EST_POINTS_NPY = (
    ROOT / "refinement" / "refinement_output" / "triangulated_points_refined_filtered.npy"
)

GT_PLY = (
    ROOT / "data" / "dataset_kicker" / "ground_truth_scan" / "scan2.ply"
)

OUT_DIR = ROOT / "evaluation" / "evaluation_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load estimated 3D points
# -----------------------------
print("Loading estimated 3D points...")
points_est = np.load(EST_POINTS_NPY)   # N x 3
pc_est = o3d.geometry.PointCloud()
pc_est.points = o3d.utility.Vector3dVector(points_est)

print(f"Estimated points: {points_est.shape[0]}")

# -----------------------------
# Load GT point cloud
# -----------------------------
print("Loading GT point cloud...")
if not GT_PLY.exists():
    raise FileNotFoundError(f"GT PLY not found: {GT_PLY}")

pc_gt = o3d.io.read_point_cloud(str(GT_PLY))
print(f"GT points: {len(pc_gt.points)}")

# -----------------------------
# Downsample (for stable ICP)
# -----------------------------
voxel = 0.05  # meter (5cm)
pc_est_ds = pc_est.voxel_down_sample(voxel)
pc_gt_ds = pc_gt.voxel_down_sample(voxel)

# -----------------------------
# ICP Alignment (point-to-point)
# -----------------------------
print("Running ICP alignment...")
threshold = 0.2  # meter
reg = o3d.pipelines.registration.registration_icp(
    pc_est_ds,
    pc_gt_ds,
    threshold,
    np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

T = reg.transformation
print("Transformation:\n", T)

pc_est.transform(T)

# -----------------------------
# Distance evaluation
# -----------------------------
print("Computing distances...")
distances = pc_est.compute_point_cloud_distance(pc_gt)
distances = np.asarray(distances)

rmse = np.sqrt(np.mean(distances ** 2))
mean_err = np.mean(distances)
median_err = np.median(distances)

# -----------------------------
# Save results
# -----------------------------
np.save(OUT_DIR / "distances.npy", distances)
o3d.io.write_point_cloud(
    str(OUT_DIR / "aligned_estimated_points.ply"), pc_est
)

with open(OUT_DIR / "summary.txt", "w") as f:
    f.write(f"Mean error (m): {mean_err:.4f}\n")
    f.write(f"Median error (m): {median_err:.4f}\n")
    f.write(f"RMSE (m): {rmse:.4f}\n")

print("\n✅ Evaluation finished")
print(f"Mean error   : {mean_err*100:.2f} cm")
print(f"Median error : {median_err*100:.2f} cm")
print(f"RMSE         : {rmse*100:.2f} cm")
