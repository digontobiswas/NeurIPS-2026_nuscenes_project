import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

TRAJECTORY_DIR = 'outputs/trajectories'
INTENT_PATH    = 'outputs/causal_graphs/agent_intents.pkl'
OUTPUT_DIR     = 'outputs/evaluation'
FIGURES        = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES,    exist_ok=True)


def compute_reward(positions, safe_distance=5.0):
    """
    Compute reward for a predicted trajectory.
    Higher = safer and smoother.
    Same philosophy as Vista reward function.
    """
    pos = np.array(positions, dtype=float)

    if len(pos) < 2:
        return {'total': 0.0, 'smoothness': 0.0, 'safety': 0.0}

    diffs      = np.diff(pos, axis=0)
    speeds     = np.sqrt(np.sum(diffs ** 2, axis=1))
    smoothness = float(1.0 / (1.0 + np.std(speeds)))

    dists_from_origin = np.sqrt(np.sum(pos ** 2, axis=1))
    safety             = float(
        np.mean(np.clip(dists_from_origin / safe_distance, 0, 1))
    )

    total = 0.5 * smoothness + 0.5 * safety

    return {
        'total'     : total,
        'smoothness': smoothness,
        'safety'    : safety
    }


traj_files = [
    f for f in os.listdir(TRAJECTORY_DIR)
    if f.endswith('.pkl')
]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

print('REWARD FUNCTION EVALUATION')
print('=' * 55)
print(f"{'Category':<25} {'Smooth':<10} {'Safety':<10} {'Total'}")
print('-' * 55)

all_rewards = []

for inst_token, traj in list(trajectories.items())[:10]:
    positions = [[p['x'], p['y']] for p in traj]
    reward    = compute_reward(positions)
    cat       = traj[0]['category'].split('.')[-1]
    all_rewards.append(reward)

    print(
        f"{cat:<25} "
        f"{reward['smoothness']:<10.4f} "
        f"{reward['safety']:<10.4f} "
        f"{reward['total']:.4f}"
    )

avg_total      = np.mean([r['total']      for r in all_rewards])
avg_smoothness = np.mean([r['smoothness'] for r in all_rewards])
avg_safety     = np.mean([r['safety']     for r in all_rewards])

print()
print(f"Average smoothness : {avg_smoothness:.4f}")
print(f"Average safety     : {avg_safety:.4f}")
print(f"Average total      : {avg_total:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Reward Function Analysis', fontsize=13)

totals      = [r['total']      for r in all_rewards]
smooths     = [r['smoothness'] for r in all_rewards]
safeties    = [r['safety']     for r in all_rewards]

axes[0].bar(range(len(totals)), totals,
            color='steelblue', alpha=0.8)
axes[0].set_title('Total Reward per Agent')
axes[0].set_xlabel('Agent index')
axes[0].set_ylabel('Reward')
axes[0].axhline(y=avg_total, color='r',
                linestyle='--', label=f'Mean={avg_total:.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].scatter(smooths, safeties,
                c=totals, cmap='RdYlGn',
                s=80, alpha=0.8)
axes[1].set_title('Smoothness vs Safety')
axes[1].set_xlabel('Smoothness')
axes[1].set_ylabel('Safety')
axes[1].grid(True, alpha=0.3)

axes[2].hist(totals, bins=10,
             color='green', alpha=0.8, edgecolor='white')
axes[2].set_title('Reward Distribution')
axes[2].set_xlabel('Total Reward')
axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(FIGURES, 'reward_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out_path}")
plt.show()
plt.close()

out_path = os.path.join(OUTPUT_DIR, 'reward_results.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(all_rewards, f)
print(f"Reward results saved: {out_path}")