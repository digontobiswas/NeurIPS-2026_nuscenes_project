import os
import pickle
import numpy as np
import torch

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/v2x_cooperative'

os.makedirs(OUTPUT_DIR, exist_ok=True)

traj_files = [
    f for f in os.listdir(TRAJECTORY_DIR)
    if f.endswith('.pkl')
]

if len(traj_files) == 0:
    print("No trajectories found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

agents = list(trajectories.keys())
print(f"Total agents available: {len(agents)}")

NUM_AGENTS  = min(5, len(agents))
sel_agents  = agents[:NUM_AGENTS]

agent_states = {}

for inst_token in sel_agents:
    traj = trajectories[inst_token]
    last = traj[-1]

    if len(traj) > 1:
        vel_x = traj[-1]['x'] - traj[-2]['x']
        vel_y = traj[-1]['y'] - traj[-2]['y']
    else:
        vel_x = 0.0
        vel_y = 0.0

    state = {
        'instance_token': inst_token,
        'position'      : [last['x'], last['y'], last['z']],
        'velocity'       : [vel_x, vel_y, 0.0],
        'category'      : last['category'],
        'n_frames'      : len(traj)
    }
    agent_states[inst_token] = state

print()
print(f"MULTI-AGENT SETUP ({NUM_AGENTS} agents):")
print('=' * 65)
print(f"{'#':<5} {'Category':<25} {'Position (x,y)':<25} {'Velocity'}")
print('-' * 65)

for i, (token, state) in enumerate(agent_states.items()):
    pos = state['position']
    vel = state['velocity']
    cat = state['category'].split('.')[-1]
    print(
        f"{i:<5} {cat:<25} "
        f"({pos[0]:.1f}, {pos[1]:.1f}){'':<10} "
        f"({vel[0]:.2f}, {vel[1]:.2f})"
    )

out_path = os.path.join(OUTPUT_DIR, 'agent_states.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(agent_states, f)

print(f"\nAgent states saved: {out_path}")