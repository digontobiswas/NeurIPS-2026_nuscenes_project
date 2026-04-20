from nuscenes.nuscenes import NuScenes
from collections import defaultdict
import numpy as np
import pickle
import os

DATAROOT   = 'D:/nuscenes_project/data/nuscenes'
OUTPUT_DIR = 'outputs/trajectories'

os.makedirs(OUTPUT_DIR, exist_ok=True)

nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)

my_scene     = nusc.scene[0]
sample_token = my_scene['first_sample_token']
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

out_path = os.path.join(OUTPUT_DIR, 'trajectories_scene0.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(dict(trajectories), f)

print(f"Saved: {out_path}")
print(f"Total unique agents: {len(trajectories)}")
print()
print(f"{'#':<5} {'Category':<35} {'Frames':<8} {'Distance(m)':<14} {'Speed(m/s)'}")
print('-' * 75)

for i, (inst_token, traj) in enumerate(trajectories.items()):
    category = traj[0]['category']
    frames   = len(traj)

    if frames > 1:
        total_dist = 0.0
        for j in range(1, frames):
            dx = traj[j]['x'] - traj[j-1]['x']
            dy = traj[j]['y'] - traj[j-1]['y']
            total_dist += np.sqrt(dx**2 + dy**2)
        dt        = (traj[-1]['timestamp'] - traj[0]['timestamp']) / 1e6
        avg_speed = total_dist / dt if dt > 0 else 0.0
    else:
        total_dist = 0.0
        avg_speed  = 0.0

    print(f"{i:<5} {category:<35} {frames:<8} {total_dist:<14.2f} {avg_speed:.2f}")

moving     = sum(1 for t in trajectories.values() if len(t) > 1 and
                 np.sqrt((t[-1]['x']-t[0]['x'])**2 + (t[-1]['y']-t[0]['y'])**2) > 1.0)
stationary = len(trajectories) - moving

print()
print('MOVING vs STATIONARY:')
print(f"  Moving    : {moving}")
print(f"  Stationary: {stationary}")