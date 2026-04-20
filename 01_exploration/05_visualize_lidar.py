from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import matplotlib.pyplot as plt
import numpy as np
import os

DATAROOT   = 'D:/nuscenes_project/data/nuscenes'
OUTPUT_DIR = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

nusc         = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
my_scene     = nusc.scene[0]
sample_token = my_scene['first_sample_token']
sample       = nusc.get('sample', sample_token)

lidar_token = sample['data']['LIDAR_TOP']
lidar_data  = nusc.get('sample_data', lidar_token)
lidar_path  = os.path.join(DATAROOT, lidar_data['filename'])

pc     = LidarPointCloud.from_file(lidar_path)
points = pc.points

print(f"LiDAR points : {points.shape[1]}")
print(f"X range      : {points[0].min():.1f} to {points[0].max():.1f} m")
print(f"Y range      : {points[1].min():.1f} to {points[1].max():.1f} m")
print(f"Z range      : {points[2].min():.1f} to {points[2].max():.1f} m")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(f"LiDAR — Scene: {my_scene['name']}", fontsize=13)

axes[0].scatter(points[0], points[1], s=0.3, c=points[2],
                cmap='viridis', alpha=0.6)
axes[0].set_title('Top-down BEV')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].set_aspect('equal')
axes[0].set_xlim(-50, 50)
axes[0].set_ylim(-50, 50)

axes[1].scatter(points[0], points[2], s=0.3, c=points[1],
                cmap='plasma', alpha=0.6)
axes[1].set_title('Front view (elevation)')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Z (m)')
axes[1].set_xlim(-50, 50)
axes[1].set_ylim(-3, 10)

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'lidar_scene0.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()