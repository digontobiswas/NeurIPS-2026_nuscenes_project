import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/figures'

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

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Agent Trajectories — nuScenes Scene 0', fontsize=14)

colors = cm.tab20(np.linspace(0, 1, len(trajectories)))

for (inst_token, traj), color in zip(trajectories.items(), colors):
    xs = [p['x'] for p in traj]
    ys = [p['y'] for p in traj]
    cat = traj[0]['category'].split('.')[-1]

    axes[0].plot(xs, ys, '-', color=color,
                 linewidth=1.5, alpha=0.8)
    axes[0].plot(xs[0], ys[0], 's', color=color, markersize=5)
    axes[0].plot(xs[-1], ys[-1], '^', color=color, markersize=5)

axes[0].set_title('All Agent Trajectories\n(■ start  ▲ end)')
axes[0].set_xlabel('X position (m)')
axes[0].set_ylabel('Y position (m)')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

cat_groups = {}
for inst_token, traj in trajectories.items():
    cat = traj[0]['category'].split('.')[0]
    if cat not in cat_groups:
        cat_groups[cat] = {'xs': [], 'ys': []}
    cat_groups[cat]['xs'].extend([p['x'] for p in traj])
    cat_groups[cat]['ys'].extend([p['y'] for p in traj])

cat_colors = {
    'vehicle': '#4C72B0',
    'human'  : '#DD8452',
    'movable': '#55A868',
    'static' : '#C44E52'
}

for cat, data in cat_groups.items():
    color = cat_colors.get(cat, '#888888')
    axes[1].scatter(
        data['xs'], data['ys'],
        c=color, s=3, alpha=0.5, label=cat
    )

axes[1].set_title('Agent Positions by Category')
axes[1].set_xlabel('X position (m)')
axes[1].set_ylabel('Y position (m)')
axes[1].set_aspect('equal')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'trajectories_full.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()