import numpy as np
import cv2
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]   # VisionPipeline/

REFINE_OUTPUT_DIR = ROOT / "refinement" / "refinement_output"
POINTS_FILE = REFINE_OUTPUT_DIR / "triangulated_points_refined_filtered.npy"
CAMERA_FILE = REFINE_OUTPUT_DIR / "camera_poses_refined.npy"

DATA_IMG_DIR = ROOT / "data" / "dataset_kicker" / "images"
GT_2D_DIR = ROOT / "data" / "dataset_kicker" / "gt_2d"

# -----------------------------
# Load data
# -----------------------------
points3d = np.load(POINTS_FILE)  # N x 3
cameras = np.load(CAMERA_FILE, allow_pickle=True).item()

print(f"Loaded {points3d.shape[0]} 3D points and {len(cameras)} cameras.")

# -----------------------------
# Camera intrinsics
# -----------------------------
K = np.array([
    [1000, 0, 1920 / 2],
    [0, 1000, 1080 / 2],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros(5)

# -----------------------------
# Reprojection error evaluation
# -----------------------------
errors = []
used_images = 0

img_files = sorted(DATA_IMG_DIR.glob("*.*"))

for img_file in img_files:
    img_name = img_file.stem

    if img_name not in cameras:
        print(f"{img_name}: camera pose not found, skip.")
        continue

    gt_file = GT_2D_DIR / f"{img_name}.npy"
    if not gt_file.exists():
        print(f"{img_name}: no 2D GT, skip.")
        continue

    cam = cameras[img_name]
    rvec = cam["rvec"]
    tvec = cam["tvec"]

    # Project 3D points
    proj_pts, _ = cv2.projectPoints(points3d, rvec, tvec, K, dist_coeffs)
    proj_pts = proj_pts.reshape(-1, 2)

    gt_pts = np.load(gt_file)

    if gt_pts.shape[0] != proj_pts.shape[0]:
        print(
            f"{img_name}: GT points ({gt_pts.shape[0]}) "
            f"!= projected points ({proj_pts.shape[0]}), skip."
        )
        continue

    img_errors = np.linalg.norm(proj_pts - gt_pts, axis=1)
    errors.extend(img_errors)
    used_images += 1

# -----------------------------
# Compute RMS reprojection error
# -----------------------------
if len(errors) > 0:
    rms_error = np.sqrt(np.mean(np.square(errors)))
else:
    rms_error = float("nan")

print("\n✅ Reprojection evaluation finished")
print(f"Used images        : {used_images}")
print(f"RMS reprojection error : {rms_error:.2f} pixels")
