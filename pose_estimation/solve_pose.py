import cv2
import numpy as np
from pathlib import Path
import sys

# =============================
# PATH SETUP (สำคัญมาก)
# =============================
ROOT = Path(__file__).resolve().parents[1]   # VisionPipeline/
sys.path.append(str(ROOT))

DATA_IMG_DIR   = ROOT / "data/dataset_kicker/images"
POINTS3D_FILE  = ROOT / "triangulation/triangulated_points.npy"
DESCS3D_FILE   = ROOT / "triangulation/triangulated_desc.npy"
OUT_POSE_DIR   = ROOT / "pose_estimation/output_pose"
OUT_POSE_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Validate input files
# =============================
if not POINTS3D_FILE.exists() or not DESCS3D_FILE.exists():
    raise FileNotFoundError("triangulated_points.npy or triangulated_desc.npy not found. Run triangulation first.")

# =============================
# Load 3D map
# =============================
points3d = np.load(POINTS3D_FILE)
desc3d   = np.load(DESCS3D_FILE)

if len(points3d) == 0:
    raise RuntimeError("3D points are empty. Triangulation failed earlier.")

print(f"[OK] Loaded {points3d.shape[0]} 3D points")

# =============================
# ORB + Matcher
# =============================
orb = cv2.ORB_create(2000)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# =============================
# Loop over images
# =============================
for img_file in sorted(DATA_IMG_DIR.glob("*")):

    img = cv2.imread(str(img_file))
    if img is None:
        print(f"[WARN] cannot read {img_file.name}, skip")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kps2d, des2d = orb.detectAndCompute(gray, None)
    if des2d is None or len(kps2d) < 10:
        print(f"[WARN] no features in {img_file.name}")
        continue

    # 3D ↔ 2D matching
    matches = bf.match(desc3d, des2d)
    matches = sorted(matches, key=lambda m: m.distance)[:300]

    if len(matches) < 6:
        print(f"[WARN] not enough matches in {img_file.name}")
        continue

    pts3d_matched = np.array([points3d[m.queryIdx] for m in matches], np.float32)
    pts2d_matched = np.array([kps2d[m.trainIdx].pt for m in matches], np.float32)

    # =============================
    # Camera intrinsics (placeholder)
    # =============================
    h, w = img.shape[:2]
    K = np.array([[1000, 0, w / 2],
                  [0, 1000, h / 2],
                  [0,    0,      1]], dtype=np.float32)
    dist = np.zeros(5)

    # =============================
    # solvePnP
    # =============================
    ok, rvec, tvec = cv2.solvePnP(
        pts3d_matched,
        pts2d_matched,
        K,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        print(f"[FAIL] solvePnP failed: {img_file.name}")
        continue

    print(f"[OK] {img_file.name}")
    print("     rvec:", rvec.ravel())
    print("     tvec:", tvec.ravel())

    # save pose
    np.save(OUT_POSE_DIR / f"{img_file.stem}_rvec.npy", rvec)
    np.save(OUT_POSE_DIR / f"{img_file.stem}_tvec.npy", tvec)

    # =============================
    # Visualization
    # =============================
    img_vis = img.copy()
    proj, _ = cv2.projectPoints(pts3d_matched, rvec, tvec, K, dist)
    for p in proj.reshape(-1, 2):
        u, v = int(p[0]), int(p[1])
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img_vis, (u, v), 2, (0, 255, 0), -1)

    cv2.imwrite(str(OUT_POSE_DIR / f"{img_file.stem}_reproj.jpg"), img_vis)

print("✅ Done. Check pose_estimation/output_pose/")
