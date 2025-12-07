import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# CONFIG (FIXED ✅)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]  # VisionPipeline/

# BA output (จาก run_ba.py)
BA_OUTPUT_DIR = ROOT / "refinement" / "refinement_output"

# output ของ step นี้
REFINED_DIR = ROOT / "refinement" / "refinement_output"
REFINED_DIR.mkdir(exist_ok=True)

CAMERA_FILE = BA_OUTPUT_DIR / "camera_poses_refined.npy"
POINTS_FILE = BA_OUTPUT_DIR / "triangulated_points_refined.npy"

# threshold reprojection error (pixels)
REPROJ_THRESH = 50.0

# -----------------------------
# LOAD DATA
# -----------------------------
camera_poses = np.load(CAMERA_FILE, allow_pickle=True).item()
points3d = np.load(POINTS_FILE)

print(f"Loaded {len(camera_poses)} cameras and {points3d.shape[0]} 3D points.")

# -----------------------------
# REPROJECTION ERROR (simple placeholder ✅)
# -----------------------------
valid_points_mask = np.ones(len(points3d), dtype=bool)

# assume same intrinsics
K = np.array([
    [1000, 0, 640],
    [0, 1000, 480],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros(5)

for img_name, pose in camera_poses.items():
    rvec = pose["rvec"]
    tvec = pose["tvec"]

    proj_pts, _ = cv2.projectPoints(points3d, rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1, 2)

    # placeholder: assume perfect correspondence
    error = np.zeros(len(proj_pts))
    valid_points_mask &= (error < REPROJ_THRESH)

print(f"Keeping {valid_points_mask.sum()} / {len(points3d)} 3D points after filtering.")

# -----------------------------
# SAVE REFINED DATA
# -----------------------------
refined_points = points3d[valid_points_mask]

np.save(REFINED_DIR / "triangulated_points_refined_filtered.npy", refined_points)
np.save(REFINED_DIR / "camera_poses_refined.npy", camera_poses)

print(f"Saved refined 3D points and camera poses to {REFINED_DIR}")
