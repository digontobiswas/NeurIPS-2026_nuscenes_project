import os
import pickle
import numpy as np

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/evaluation'

os.makedirs(OUTPUT_DIR, exist_ok=True)

traj_files = [
    f for f in os.listdir(TRAJECTORY_DIR)
    if f.endswith('.pkl')
]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)


def detect_corner_cases(trajectories, speed_thresh=5.0,
                        acc_thresh=2.0, prox_thresh=5.0):
    """
    Detect corner cases in trajectories:
    - High speed agents
    - High acceleration agents
    - Near-collision events (agents very close)
    """
    corner_cases = []
    agents       = list(trajectories.keys())

    for inst_token, traj in trajectories.items():
        cat = traj[0]['category']

        if len(traj) < 3:
            continue

        positions = np.array([[p['x'], p['y']] for p in traj])
        times     = np.array([p['timestamp'] for p in traj]) / 1e6

        diffs  = np.diff(positions, axis=0)
        dt     = np.diff(times)
        dt     = np.where(dt > 0, dt, 1e-6)

        speeds = np.sqrt(np.sum(diffs ** 2, axis=1)) / dt
        accels = np.abs(np.diff(speeds)) / dt[1:]

        max_speed = float(np.max(speeds))
        max_accel = float(np.max(accels)) if len(accels) > 0 else 0.0

        if max_speed > speed_thresh:
            corner_cases.append({
                'type'    : 'high_speed',
                'instance': inst_token,
                'category': cat,
                'value'   : max_speed,
                'unit'    : 'm/s'
            })

        if max_accel > acc_thresh:
            corner_cases.append({
                'type'    : 'high_acceleration',
                'instance': inst_token,
                'category': cat,
                'value'   : max_accel,
                'unit'    : 'm/s²'
            })

    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            traj_i  = trajectories[agents[i]]
            traj_j  = trajectories[agents[j]]
            min_len = min(len(traj_i), len(traj_j))

            if min_len == 0:
                continue

            for k in range(min_len):
                dx = traj_i[k]['x'] - traj_j[k]['x']
                dy = traj_i[k]['y'] - traj_j[k]['y']
                dist = np.sqrt(dx**2 + dy**2)

                if dist < prox_thresh:
                    corner_cases.append({
                        'type'     : 'near_collision',
                        'agent_1'  : agents[i][:8],
                        'agent_2'  : agents[j][:8],
                        'category1': trajectories[agents[i]][0]['category'],
                        'category2': trajectories[agents[j]][0]['category'],
                        'value'    : float(dist),
                        'unit'     : 'm',
                        'frame'    : k
                    })
                    break

    return corner_cases


corner_cases = detect_corner_cases(trajectories)

print(f"Total corner cases detected: {len(corner_cases)}")
print()

from collections import Counter
types = Counter(cc['type'] for cc in corner_cases)

print('CORNER CASE TYPES:')
print('-' * 40)
for cc_type, count in types.items():
    print(f"  {cc_type:<25} {count}")

print()
print('CORNER CASE DETAILS:')
print('-' * 65)

for cc in corner_cases[:10]:
    if cc['type'] == 'near_collision':
        print(
            f"  [{cc['type']}] "
            f"{cc['agent_1']} ↔ {cc['agent_2']} "
            f"dist={cc['value']:.2f}m at frame {cc['frame']}"
        )
    else:
        print(
            f"  [{cc['type']}] "
            f"{cc['instance'][:8]}... "
            f"{cc['category'].split('.')[-1]} "
            f"= {cc['value']:.2f} {cc['unit']}"
        )

out_path = os.path.join(OUTPUT_DIR, 'corner_cases.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(corner_cases, f)
print(f"\nCorner cases saved: {out_path}")