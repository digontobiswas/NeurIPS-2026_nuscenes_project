import os
import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

print("CausalCoop-WM LiDAR Visualization")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

scene_index = 0
sample_index = 0
scene = nusc.scene[scene_index]
sample_token = scene["first_sample_token"]
for _ in range(sample_index):
    sample = nusc.get("sample", sample_token)
    sample_token = sample["next"]
sample = nusc.get("sample", sample_token)

lidar_token = sample["data"]["LIDAR_TOP"]
lidar_data = nusc.get("sample_data", lidar_token)
points = np.fromfile(lidar_data["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].scatter(points[:, 0], points[:, 1], s=0.3, c=points[:, 2], cmap="viridis", alpha=0.6)
axes[0].set_title("Top-down view (BEV)")
axes[0].set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].set_aspect("equal")
axes[0].set_xlim(-50, 50)
axes[0].set_ylim(-50, 50)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(points[:, 0], points[:, 2], s=0.3, c=points[:, 1], cmap="plasma", alpha=0.6)
axes[1].set_title("Front view (elevation)")
axes[1].set_xlabel("X (m)")
axes[1].set_ylabel("Z (m)")
axes[1].set_aspect("equal")
axes[1].set_xlim(-50, 50)
axes[1].set_ylim(-3, 10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "lidar_scene" + scene["token"][:8] + "_sample" + str(sample_index) + ".png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("LiDAR visualization saved to " + out_path)
print("LiDAR visualization completed.")