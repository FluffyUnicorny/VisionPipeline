import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares
import time

# ----------------------------- CONFIG -----------------------------
ROOT = Path(__file__).resolve().parents[1]   # VisionPipeline/

DATA_IMG_DIR = ROOT / "data" / "dataset_kicker" / "images"
POINTS3D_FILE = ROOT / "triangulation" / "triangulated_points.npy"
DESCS3D_FILE = ROOT / "triangulation" / "triangulated_desc.npy"
CAMERA_POSES_FILE = ROOT / "pose_estimation" / "camera_poses.npy"

# ✅ FIX: ให้ output ไปตรงที่ refinement ใช้
OUT_DIR = ROOT / "refinement" / "refinement_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_POINTS = OUT_DIR / "triangulated_points_refined.npy"
OUT_CAMERA_POSES = OUT_DIR / "camera_poses_refined.npy"

# Camera intrinsics
FX = 1000.0
FY = 1000.0
USE_IMAGE_CENTER_PRINCIPAL_POINT = True

# BA optimizer settings
MAX_NFEV = 200
LOSS = 'huber'

# Match settings
MAX_MATCHES_PER_IMAGE = 1000
MIN_MATCHES_FOR_IMAGE = 6

# ----------------------------- Helper functions -----------------------------
def pack_params(rvecs, tvecs, points3d):
    return np.hstack((rvecs.ravel(), tvecs.ravel(), points3d.ravel()))

def unpack_params(x, n_cameras, n_points):
    cam_r_vecs = x[: 3 * n_cameras].reshape((n_cameras, 3))
    cam_t_vecs = x[3 * n_cameras: 6 * n_cameras].reshape((n_cameras, 3))
    pts = x[6 * n_cameras:].reshape((n_points, 3))
    return cam_r_vecs, cam_t_vecs, pts

def project_point(rvec, tvec, point3d, K):
    R, _ = cv2.Rodrigues(rvec)
    Xc = R.dot(point3d) + tvec
    x = Xc[0] / Xc[2]
    y = Xc[1] / Xc[2]
    uv = K.dot(np.array([x, y, 1.0]))
    return uv[0], uv[1]

def reprojection_residuals(x, n_cameras, n_points,
                           camera_indices, point_indices, points_2d, Ks):
    rvecs, tvecs, pts3d = unpack_params(x, n_cameras, n_points)
    residuals = np.zeros((len(camera_indices) * 2,), dtype=float)

    for i, (ci, pi) in enumerate(zip(camera_indices, point_indices)):
        u_obs, v_obs = points_2d[i]
        K = Ks[ci]

        try:
            u_proj, v_proj = project_point(
                rvecs[ci], tvecs[ci], pts3d[pi], K
            )
        except Exception:
            u_proj, v_proj = 1e6, 1e6

        residuals[2*i] = u_proj - u_obs
        residuals[2*i+1] = v_proj - v_obs

    return residuals

# ----------------------------- Main -----------------------------
def main():
    print("Loading data...")

    for p in [POINTS3D_FILE, DESCS3D_FILE, CAMERA_POSES_FILE, DATA_IMG_DIR]:
        if not p.exists():
            raise FileNotFoundError(p)

    points3d = np.load(POINTS3D_FILE)
    desc3d = np.load(DESCS3D_FILE)
    camera_poses = np.load(CAMERA_POSES_FILE, allow_pickle=True).item()
    image_files = list(DATA_IMG_DIR.glob("*.*"))

    camera_names = sorted(camera_poses.keys())
    n_cameras = len(camera_names)
    n_points = points3d.shape[0]

    rvecs_init = np.array(
        [camera_poses[n]["rvec"].ravel() for n in camera_names]
    )
    tvecs_init = np.array(
        [camera_poses[n]["tvec"].ravel() for n in camera_names]
    )

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    cam_idx, pt_idx, pts2d, Ks = [], [], [], []

    for i, name in enumerate(camera_names):
        imgs = [p for p in image_files if p.stem == name]
        if not imgs:
            Ks.append(np.array([[FX,0,0],[0,FY,0],[0,0,1]]))
            continue

        img = cv2.imread(str(imgs[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = orb.detectAndCompute(gray, None)
        if des is None:
            Ks.append(np.array([[FX,0,0],[0,FY,0],[0,0,1]]))
            continue

        cx, cy = img.shape[1]/2, img.shape[0]/2
        Ks.append(np.array([[FX,0,cx],[0,FY,cy],[0,0,1]]))

        matches = bf.match(desc3d.astype(np.uint8), des)
        matches = sorted(matches, key=lambda m: m.distance)[:MAX_MATCHES_PER_IMAGE]

        for m in matches:
            cam_idx.append(i)
            pt_idx.append(m.queryIdx)
            pts2d.append(kps[m.trainIdx].pt)

    cam_idx = np.array(cam_idx)
    pt_idx = np.array(pt_idx)
    pts2d = np.array(pts2d)

    X0 = pack_params(rvecs_init, tvecs_init, points3d)

    print("Running Bundle Adjustment...")
    res = least_squares(
        reprojection_residuals,
        X0,
        args=(n_cameras, n_points, cam_idx, pt_idx, pts2d, Ks),
        loss=LOSS,
        max_nfev=MAX_NFEV,
        verbose=2
    )

    rvecs_opt, tvecs_opt, pts3d_opt = unpack_params(res.x, n_cameras, n_points)

    cam_refined = {
        name: {
            "rvec": rvecs_opt[i].reshape(3,1),
            "tvec": tvecs_opt[i].reshape(3,1)
        }
        for i, name in enumerate(camera_names)
    }

    np.save(OUT_CAMERA_POSES, cam_refined)
    np.save(OUT_POINTS, pts3d_opt)

    print("✅ BA finished")
    print("Saved:", OUT_CAMERA_POSES)
    print("Saved:", OUT_POINTS)

if __name__ == "__main__":
    main()
