from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLMAP_TXT = PROJECT_ROOT / "sfm" / "colmap_output" / "sparse_txt"

# COLMAP_TXT = Path("colmap_output/sparse_txt")

def read_cameras():
    cameras = {}
    with open(COLMAP_TXT / "cameras.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            cam_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


def read_images():
    images = {}
    with open(COLMAP_TXT / "images.txt") as f:
        lines = f.readlines()

    for i in range(0, len(lines), 2):
        if lines[i].startswith("#"):
            continue
        elems = lines[i].strip().split()
        img_id = int(elems[0])
        q = np.array(list(map(float, elems[1:5])))  # qw qx qy qz
        t = np.array(list(map(float, elems[5:8]))) # tx ty tz
        cam_id = int(elems[8])
        name = elems[9]

        images[img_id] = {
            "q": q,
            "t": t,
            "camera_id": cam_id,
            "name": name
        }
    return images


def read_points3D():
    points = {}
    with open(COLMAP_TXT / "points3D.txt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elems = line.strip().split()
            pid = int(elems[0])
            xyz = np.array(list(map(float, elems[1:4])))
            rgb = np.array(list(map(int, elems[4:7])))
            error = float(elems[7])
            points[pid] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error
            }
    return points

def qvec2rotmat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])

def get_intrinsic_matrix(cam):
    p = cam["params"]
    if cam["model"] == "SIMPLE_RADIAL":
        f, cx, cy, _ = p
    elif cam["model"] == "PINHOLE":
        fx, fy, cx, cy = p
        f = fx
    else:
        raise NotImplementedError(cam["model"])

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    return K

def load_ply(filename):
    points = []
    with open(filename, 'r') as f:
        header = True
        for line in f:
            if header:
                if line.strip() == "end_header":
                    header = False
                continue
            x, y, z = map(float, line.strip().split())
            points.append([x, y, z])
    return np.array(points)

if __name__ == "__main__":
    cameras = read_cameras()
    images = read_images()
    points = read_points3D()

    img_ids = list(images.keys())[:2]

    for img_id in img_ids:
        img = images[img_id]
        cam = cameras[img["camera_id"]]

        R = qvec2rotmat(img["q"])
        t = img["t"]
        C = -R.T @ t
        K = get_intrinsic_matrix(cam)

        print(f"\nImage: {img['name']}")
        print("Camera center:", C)
        print("Intrinsic K:\n", K)

# ชุด kicker มี ~31 ภาพ แต่บางภาพอาจถูก COLMAP คัดออกเพราะ match ไม่พอ