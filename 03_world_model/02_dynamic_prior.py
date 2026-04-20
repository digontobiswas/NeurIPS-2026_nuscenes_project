import os
import pickle
import numpy as np
import torch

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/world_model_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_dynamic_priors(traj, n_priors=3):
    """
    Compute position, velocity and acceleration priors
    from a trajectory — same 3-prior approach as Vista.
    """
    positions = np.array([[p['x'], p['y'], p['z']] for p in traj])

    if len(positions) < 2:
        velocity     = np.zeros(3)
        acceleration = np.zeros(3)
    else:
        velocity = positions[-1] - positions[-2]

        if len(positions) >= 3:
            vel_prev     = positions[-2] - positions[-3]
            acceleration = velocity - vel_prev
        else:
            acceleration = np.zeros(3)

    priors = {
        'position'    : positions[-1].tolist(),
        'velocity'    : velocity.tolist(),
        'acceleration': acceleration.tolist(),
        'n_frames'    : len(positions)
    }

    return priors


traj_files = [
    f for f in os.listdir(TRAJECTORY_DIR)
    if f.endswith('.pkl')
]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

all_priors = {}

print(f"Computing dynamic priors for {len(trajectories)} agents...")
print()
print(f"{'Agent':<12} {'Category':<25} {'Speed(m/s)':<12} {'Accel'}")
print('-' * 65)

for inst_token, traj in trajectories.items():
    prior = compute_dynamic_priors(traj)
    all_priors[inst_token] = prior

    vel    = np.array(prior['velocity'])
    acc    = np.array(prior['acceleration'])
    speed  = float(np.linalg.norm(vel))
    accel  = float(np.linalg.norm(acc))
    cat    = traj[0]['category'].split('.')[-1]

    print(f"{inst_token[:10]:<12} {cat:<25} {speed:<12.3f} {accel:.3f}")

out_path = os.path.join(OUTPUT_DIR, 'dynamic_priors.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(all_priors, f)

print(f"\nDynamic priors saved: {out_path}")

speeds = [
    np.linalg.norm(p['velocity'])
    for p in all_priors.values()
]
print(f"\nSpeed statistics:")
print(f"  Mean  : {np.mean(speeds):.3f} m/s")
print(f"  Max   : {np.max(speeds):.3f} m/s")
print(f"  Min   : {np.min(speeds):.3f} m/s")
print(f"  Std   : {np.std(speeds):.3f} m/s")