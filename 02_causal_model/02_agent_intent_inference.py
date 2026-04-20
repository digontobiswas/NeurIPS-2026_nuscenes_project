import os
import pickle
import numpy as np
from collections import Counter

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/causal_graphs'
INTENT_PATH    = os.path.join(OUTPUT_DIR, 'agent_intents.pkl')

os.makedirs(OUTPUT_DIR, exist_ok=True)

traj_files = [f for f in os.listdir(TRAJECTORY_DIR) if f.endswith('.pkl')]

if len(traj_files) == 0:
    print("No trajectory files found. Run 03_extract_trajectories.py first.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

intent_data = {}

for inst_token, traj in trajectories.items():
    if len(traj) < 2:
        intent = 'stationary'
    else:
        dx_total = traj[-1]['x'] - traj[0]['x']
        dy_total = traj[-1]['y'] - traj[0]['y']
        dist     = np.sqrt(dx_total**2 + dy_total**2)

        if dist < 1.0:
            intent = 'stationary'
        elif abs(dy_total) < 2.0:
            intent = 'moving_straight'
        elif dy_total > 2.0:
            intent = 'turning_left'
        elif dy_total < -2.0:
            intent = 'turning_right'
        else:
            intent = 'stopping'

    intent_data[inst_token] = {
        'intent'  : intent,
        'category': traj[0]['category'],
        'frames'  : len(traj)
    }

with open(INTENT_PATH, 'wb') as f:
    pickle.dump(intent_data, f)

print(f"Agent intents saved: {INTENT_PATH}")
print(f"Total agents       : {len(intent_data)}")
print()

counts = Counter(v['intent'] for v in intent_data.values())
print('INTENT DISTRIBUTION:')
print('-' * 35)
for intent, count in sorted(counts.items(), key=lambda x: -x[1]):
    bar = '█' * count
    print(f"  {intent:<20} {count:>3}  {bar}")

print()
print('SAMPLE AGENT INTENTS:')
print(f"{'Token':<12} {'Category':<30} {'Intent':<20} {'Frames'}")
print('-' * 75)
for i, (token, data) in enumerate(list(intent_data.items())[:10]):
    print(f"{token[:10]:<12} {data['category']:<30} {data['intent']:<20} {data['frames']}")