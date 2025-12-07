# wireframe_triangulate.py  -- cleaned multi-view pairwise triangulation (fixed)
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations
import sys

# allow running from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from shared.colmap_io import read_cameras, read_images, read_points3D, qvec2rotmat, get_intrinsic_matrix

BASE = Path("colmap_output/sparse_txt")
DATA_IMG_DIR = Path("../data/dataset_kicker/images")
OUT_PLY = Path("triangulated_wireframe.ply")
REPROJ_THRESH = 20.0

def projection_from_K_R_t(K, R, t):
    Rt = np.hstack((R, t.reshape(3,1)))
    return K.dot(Rt)

def reproj_error(P, X, uv):
    Xh = np.hstack((X,1.0))
    uvw = P.dot(Xh)
    u = uvw[0]/uvw[2]; v = uvw[1]/uvw[2]
    return np.linalg.norm(np.array([u,v]) - uv)

def extract_endpoints(lines):
    eps = []
    if lines is None: return np.zeros((0,2)), lines
    for l in lines:
        x1,y1,x2,y2 = l[0]
        eps.append((x1,y1)); eps.append((x2,y2))
    pts = np.array(eps).reshape(-1,2)
    return pts, lines

def pts_to_keypoints(pts):
    kps = []
    for (x, y) in pts:
        kps.append(cv2.KeyPoint(float(x), float(y), 31))
    return kps

def write_ply(points, filename):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def safe_imread(p: Path):
    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARN] failed to load image: {p}")
    return img

def main():
    # --- load colmap parsed data
    cameras = read_cameras()
    images = read_images()
    points3D = read_points3D()

    if len(images) == 0:
        print("No images found in COLMAP parsed output. Exiting.")
        return

    # pick N images to fuse (tune N as needed)
    img_ids = list(images.keys())[:5]

    # prepare detectors/descriptors/matcher (reuse)
    lsd = cv2.createLineSegmentDetector()
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # quick sanity: project some colmap 3D points onto first two images and save overlays
    if len(img_ids) >= 2:
        id_a, id_b = img_ids[0], img_ids[1]
        img_a = images[id_a]; img_b = images[id_b]
        cam_a = cameras[img_a["camera_id"]]; cam_b = cameras[img_b["camera_id"]]
        K_a = get_intrinsic_matrix(cam_a); K_b = get_intrinsic_matrix(cam_b)
        R_a = qvec2rotmat(img_a["q"]); t_a = img_a["t"]
        R_b = qvec2rotmat(img_b["q"]); t_b = img_b["t"]
        I_a = safe_imread(DATA_IMG_DIR / img_a["name"])
        I_b = safe_imread(DATA_IMG_DIR / img_b["name"])

        if I_a is None or I_b is None:
            print("[INFO] Skipping projection sanity check because one or both images failed to load.")
        else:
            I_a_vis = I_a.copy(); I_b_vis = I_b.copy()

            def project_point_vis(K, R, t, X):
                Xh = np.hstack((X,1.0))
                uvw = K.dot(np.hstack((R, t.reshape(3,1))).dot(Xh))
                u = uvw[0]/uvw[2]; v = uvw[1]/uvw[2]
                return np.array([u, v])

            for pid in list(points3D.keys())[:500]:
                X = points3D[pid]["xyz"]
                try:
                    uv1 = project_point_vis(K_a, R_a, t_a, X)
                    uv2 = project_point_vis(K_b, R_b, t_b, X)
                except Exception:
                    continue
                u1,v1 = int(round(uv1[0])), int(round(uv1[1]))
                u2,v2 = int(round(uv2[0])), int(round(uv2[1]))
                if 0<=u1<I_a_vis.shape[1] and 0<=v1<I_a_vis.shape[0]:
                    cv2.circle(I_a_vis,(u1,v1),2,(0,255,0),-1)
                if 0<=u2<I_b_vis.shape[1] and 0<=v2<I_b_vis.shape[0]:
                    cv2.circle(I_b_vis,(u2,v2),2,(0,255,0),-1)
            cv2.imwrite("proj_overlay_img1.jpg", I_a_vis)
            cv2.imwrite("proj_overlay_img2.jpg", I_b_vis)
            print("Wrote proj_overlay_img1.jpg / proj_overlay_img2.jpg (sanity check)")

    # --- pairwise triangulation and fusion
    all_3d_points = []

    for id1, id2 in combinations(img_ids, 2):
        img1 = images[id1]; img2 = images[id2]
        cam1 = cameras[img1["camera_id"]]; cam2 = cameras[img2["camera_id"]]
        K1 = get_intrinsic_matrix(cam1); K2 = get_intrinsic_matrix(cam2)
        R1 = qvec2rotmat(img1["q"]); t1 = img1["t"]
        R2 = qvec2rotmat(img2["q"]); t2 = img2["t"]
        P1 = projection_from_K_R_t(K1, R1, t1)
        P2 = projection_from_K_R_t(K2, R2, t2)

        # load images for this pair
        I1_pair = safe_imread(DATA_IMG_DIR / img1["name"])
        I2_pair = safe_imread(DATA_IMG_DIR / img2["name"])
        if I1_pair is None or I2_pair is None:
            print(f"[WARN] Skipping pair ({img1['name']}, {img2['name']}) due to missing image.")
            continue
        I1g = cv2.cvtColor(I1_pair, cv2.COLOR_BGR2GRAY)
        I2g = cv2.cvtColor(I2_pair, cv2.COLOR_BGR2GRAY)

        # detect endpoints
        lines1 = lsd.detect(I1g)[0]; lines2 = lsd.detect(I2g)[0]
        pts1_eps, _ = extract_endpoints(lines1); pts2_eps, _ = extract_endpoints(lines2)
        if pts1_eps.shape[0] == 0 or pts2_eps.shape[0] == 0:
            continue

        # limit endpoints for speed/noise
        pts1_eps = pts1_eps[:3000]; pts2_eps = pts2_eps[:3000]

        # compute descriptors at endpoints
        kps1 = pts_to_keypoints(pts1_eps); kps2 = pts_to_keypoints(pts2_eps)
        kps1, des1 = orb.compute(I1g, kps1); kps2, des2 = orb.compute(I2g, kps2)
        if des1 is None or des2 is None:
            continue

        # match and keep top matches
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)[:500]
        if len(matches) < 8:
            continue

        matched_pts1 = np.array([pts1_eps[m.queryIdx] for m in matches])
        matched_pts2 = np.array([pts2_eps[m.trainIdx] for m in matches])

        pts1_t = matched_pts1.T.astype(np.float64); pts2_t = matched_pts2.T.astype(np.float64)

        # triangulate and filter
        pts4 = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
        pts3d = (pts4[:3] / pts4[3]).T

        for i, X in enumerate(pts3d):
            Z1 = (R1.dot(X) + t1)[2]; Z2 = (R2.dot(X) + t2)[2]
            if Z1 <= 0 or Z2 <= 0:
                continue
            if reproj_error(P1, X, matched_pts1[i]) > REPROJ_THRESH:
                continue
            if reproj_error(P2, X, matched_pts2[i]) > REPROJ_THRESH:
                continue
            all_3d_points.append(X)

    # finalize & save
    if len(all_3d_points) == 0:
        print("Total fused 3D points: 0 (nothing to save).")
        # still create an empty ply (optional) or skip
        write_ply([], OUT_PLY)
        print("Wrote empty", OUT_PLY)
    else:
        all_3d_points = np.array(all_3d_points)
        print("Total fused 3D points:", all_3d_points.shape[0])
        write_ply(all_3d_points, OUT_PLY)
        print("Wrote", OUT_PLY)

        # สร้าง dummy descriptors สำหรับ 3D points (ถ้าต้องการ)
        descriptors3d = np.random.randint(0,256,(all_3d_points.shape[0],32),dtype=np.uint8)
        OUT_DIR = Path(__file__).parent

        np.save(OUT_DIR / "triangulated_points.npy", all_3d_points)
        np.save(OUT_DIR / "triangulated_desc.npy", descriptors3d)

        print("Saved triangulated_points.npy and triangulated_desc.npy")

if __name__ == "__main__":
    main()
