import time
import subprocess
import sys

def run(cmd):
    t0 = time.time()
    print(f"\n▶ {cmd}")
    subprocess.run([sys.executable] + cmd.split(), check=True)
    print(f"⏱ {time.time() - t0:.1f} sec")

pipeline = [
    "sfm/run_sfm.py",
    "triangulation/triangulate.py",
    "pose_estimation/solve_pose.py",
    "pose_estimation/refine_pose.py",
    "pose_estimation/collect_camera_poses.py",
    "bundle_adjustment/run_ba.py",
    "refinement/refine_structure.py",
    "evaluation/evaluate.py",
    "evaluation/eval_align_and_localize.py"
]

for p in pipeline:
    run(p)
