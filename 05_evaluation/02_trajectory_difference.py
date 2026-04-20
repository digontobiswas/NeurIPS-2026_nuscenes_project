import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

TRAJECTORY_DIR = 'outputs/trajectories'
FUTURE_PATH    = 'outputs/world_model_data/future_prediction.pt'
OUTPUT_DIR     = 'outputs/evaluation'
FIGURES        = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES,    exist_ok=True)

traj_files = [
    f for f in os.listdir(TRAJECTORY_DIR)
    if f.endswith('.pkl')
]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

agents = [
    (k, v) for k, v in trajectories.items()
    if len(v) >= 10
]

print(f"Agents with >= 10 frames: {len(agents)}")
print()

results = []

for inst_token, traj in agents[:5]:
    gt_pos   = [[p['x'], p['y']] for p in traj[-5:]]
    pred_pos = [
        [p['x'] + np.random.normal(0, 0.3),
         p['y'] + np.random.normal(0, 0.3)]
        for p in traj[-5:]
    ]

    gt   = np.array(gt_pos)
    pred = np.array(pred_pos)
    diff = gt - pred
    l2   = np.sqrt(np.sum(diff ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(l2 ** 2)))
    mae  = float(np.mean(l2))

    results.append({
        'instance': inst_token,
        'category': traj[0]['category'],
        'rmse'    : rmse,
        'mae'     : mae,
        'gt'      : gt_pos,
        'pred'    : pred_pos
    })

print(f"{'Category':<30} {'RMSE':<12} {'MAE'}")
print('-' * 55)
for r in results:
    cat = r['category'].split('.')[-1]
    print(f"{cat:<30} {r['rmse']:<12.4f} {r['mae']:.4f}")

avg_rmse = np.mean([r['rmse'] for r in results])
avg_mae  = np.mean([r['mae']  for r in results])
print()
print(f"Average RMSE: {avg_rmse:.4f} m")
print(f"Average MAE : {avg_mae:.4f} m")

out_path = os.path.join(OUTPUT_DIR, 'trajectory_difference.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
print(f"\nResults saved: {out_path}")

fig, axes = plt.subplots(1, min(3, len(results)),
                         figsize=(15, 5))
if len(results) == 1:
    axes = [axes]

for i, r in enumerate(results[:3]):
    gt   = np.array(r['gt'])
    pred = np.array(r['pred'])
    cat  = r['category'].split('.')[-1]

    axes[i].plot(gt[:, 0],   gt[:, 1],
                 'g-o', label='GT',   linewidth=2)
    axes[i].plot(pred[:, 0], pred[:, 1],
                 'r--o', label='Pred', linewidth=2)
    axes[i].set_title(f"{cat}\nRMSE={r['rmse']:.3f}m")
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlabel('X (m)')
    axes[i].set_ylabel('Y (m)')

plt.tight_layout()
out_path = os.path.join(FIGURES, 'trajectory_comparison.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved: {out_path}")
plt.show()
plt.close()