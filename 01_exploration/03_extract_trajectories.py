from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import numpy as np

DATAROOT = 'D:/nuscenes_project/data/nuscenes'

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

# ── Extract trajectories for all agents in scene 0 ───────────
my_scene     = nusc.scene[0]
sample_token = my_scene['first_sample_token']

# instance_token → list of position dicts
trajectories = defaultdict(list)

while sample_token:
    sample = nusc.get('sample', sample_token)
    for ann_token in sample['anns']:
        ann  = nusc.get('sample_annotation', ann_token)
        x, y, z = ann['translation']
        trajectories[ann['instance_token']].append({
            'x'        : x,
            'y'        : y,
            'z'        : z,
            'timestamp': sample['timestamp'],
            'category' : ann['category_name'],
            'num_lidar': ann['num_lidar_pts']
        })
    sample_token = sample['next']

# ── Print trajectory summary ──────────────────────────────────
print(f"Total unique agents: {len(trajectories)}")
print()
print(f"{'#':<5} {'Category':<35} {'Frames':<8} {'Distance(m)':<14} {'Avg Speed(m/s)'}")
print('-' * 75)

for i, (inst_token, traj) in enumerate(trajectories.items()):
    category = traj[0]['category']
    frames   = len(traj)

    if frames > 1:
        # Total path distance
        total_dist = 0.0
        for j in range(1, frames):
            dx = traj[j]['x'] - traj[j-1]['x']
            dy = traj[j]['y'] - traj[j-1]['y']
            total_dist += np.sqrt(dx**2 + dy**2)

        # Time difference in seconds (timestamps in microseconds)
        dt = (traj[-1]['timestamp'] - traj[0]['timestamp']) / 1e6
        avg_speed = total_dist / dt if dt > 0 else 0.0
    else:
        total_dist = 0.0
        avg_speed  = 0.0

    print(f"{i:<5} {category:<35} {frames:<8} {total_dist:<14.2f} {avg_speed:.2f}")

# ── Identify moving vs stationary agents ──────────────────────
print()
print('MOVING vs STATIONARY:')
print('-' * 40)
moving     = 0
stationary = 0

for traj in trajectories.values():
    if len(traj) > 1:
        dx = traj[-1]['x'] - traj[0]['x']
        dy = traj[-1]['y'] - traj[0]['y']
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 1.0:   # moved more than 1 metre
            moving += 1
        else:
            stationary += 1

print(f"  Moving agents    : {moving}")
print(f"  Stationary agents: {stationary}")