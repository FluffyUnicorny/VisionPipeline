import cv2
import numpy as np
from pathlib import Path
import sys

# -----------------------------
# PATH SETUP (✅ สำคัญมาก)
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]   # VisionPipeline/
sys.path.append(str(ROOT))

DATA_IMG_DIR  = ROOT / "data/dataset_kicker/images"
POINTS3D_FILE = ROOT / "triangulation/triangulated_points.npy"
DESCS3D_FILE  = ROOT / "triangulation/triangulated_desc.npy"

IN_POSE_DIR   = ROOT / "pose_estimation/output_pose"
OUT_POSE_DIR  = ROOT / "pose_estimation/output_refined_pose"
OUT_POSE_DIR.mkdir(parents=True, exist_ok=True)


# โหลด 3D points + descriptors
points3d = np.load(POINTS3D_FILE)
desc3d = np.load(DESCS3D_FILE)

# ORB detector + matcher
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

for img_file in DATA_IMG_DIR.glob("*.*"):
    img = cv2.imread(str(img_file))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps2d, des2d = orb.detectAndCompute(gray, None)
    if des2d is None or len(kps2d)==0:
        continue

    # match descriptors
    matches = bf.match(desc3d, des2d)
    matches = sorted(matches, key=lambda m: m.distance)[:500]
    if len(matches)<4:
        continue

    pts3d_matched = np.array([points3d[m.queryIdx] for m in matches], dtype=np.float32)
    pts2d_matched = np.array([kps2d[m.trainIdx].pt for m in matches], dtype=np.float32)

    # โหลด rvec/tvec เดิม
    rvec = np.load(IN_POSE_DIR / f"{img_file.stem}_rvec.npy")
    tvec = np.load(IN_POSE_DIR / f"{img_file.stem}_tvec.npy")

    K = np.array([[1000, 0, img.shape[1]/2],
                  [0, 1000, img.shape[0]/2],
                  [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5)

    # -----------------------------
    # Refine rvec/tvec
    # -----------------------------
    rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
        objectPoints=pts3d_matched,
        imagePoints=pts2d_matched,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 1e-6)
    )

    # save refined pose
    np.save(OUT_POSE_DIR / f"{img_file.stem}_rvec.npy", rvec_refined)
    np.save(OUT_POSE_DIR / f"{img_file.stem}_tvec.npy", tvec_refined)

    # visualize reprojected points
    img_vis = img.copy()
    proj_pts, _ = cv2.projectPoints(pts3d_matched, rvec_refined, tvec_refined, K, dist_coeffs)
    for uv in proj_pts:
        uv = uv.ravel()
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
            cv2.circle(img_vis, (u,v), 3, (0,255,0), -1)
    cv2.imwrite(OUT_POSE_DIR / f"{img_file.stem}_reproj.jpg", img_vis)

print("Bundle Adjustment (Refine Pose) done. Check output_refined_pose folder.")
