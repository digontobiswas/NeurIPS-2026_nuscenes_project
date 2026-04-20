import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUTPUT_DIR = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


trajectories  = load_if_exists('outputs/trajectories/trajectories_scene0.pkl')
intent_data   = load_if_exists('outputs/causal_graphs/agent_intents.pkl')
corner_cases  = load_if_exists('outputs/evaluation/corner_cases.pkl')
reward_data   = load_if_exists('outputs/evaluation/reward_results.pkl')
fid_fvd       = load_if_exists('outputs/evaluation/fid_fvd_results.pkl')
bandwidth     = load_if_exists('outputs/v2x_cooperative/bandwidth_ablation.pkl')

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    'CausalCoop-WM — Full Pipeline Summary\n'
    'NeurIPS 2026 Research Project',
    fontsize=15, fontweight='bold'
)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :2])
if trajectories:
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(trajectories)))
    for (token, traj), color in zip(list(trajectories.items())[:15], colors):
        xs = [p['x'] for p in traj]
        ys = [p['y'] for p in traj]
        ax1.plot(xs, ys, '-', color=color, linewidth=1, alpha=0.7)
ax1.set_title('Agent Trajectories', fontsize=11)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.2)

ax2 = fig.add_subplot(gs[0, 2:])
if intent_data:
    from collections import Counter
    intents = Counter(v['intent'] for v in intent_data.values())
    labels  = list(intents.keys())
    sizes   = list(intents.values())
    colors2 = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2']
    ax2.pie(sizes, labels=labels, colors=colors2[:len(labels)],
            autopct='%1.1f%%', startangle=90)
ax2.set_title('Intent Distribution (PS-1)', fontsize=11)

ax3 = fig.add_subplot(gs[1, :2])
if bandwidth:
    ratios = [r * 100 for r in bandwidth['bandwidth_ratios']]
    errors = bandwidth['errors']
    ax3.plot(ratios, errors, 'b-o', linewidth=2, markersize=6)
    ax3.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
    ax3.legend(fontsize=9)
ax3.set_title('V2X Bandwidth Ablation (PS-2)', fontsize=11)
ax3.set_xlabel('Bandwidth used (%)')
ax3.set_ylabel('MSE Error')
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[1, 2:])
if reward_data:
    totals = [r['total'] for r in reward_data]
    ax4.bar(range(len(totals)), totals,
            color='green', alpha=0.8)
    ax4.axhline(y=np.mean(totals), color='r',
                linestyle='--',
                label=f'Mean={np.mean(totals):.3f}')
    ax4.legend(fontsize=9)
ax4.set_title('Reward Function Results', fontsize=11)
ax4.set_xlabel('Agent')
ax4.set_ylabel('Reward')
ax4.grid(True, alpha=0.3, axis='y')

ax5 = fig.add_subplot(gs[2, :2])
if corner_cases:
    from collections import Counter
    cc_types = Counter(cc['type'] for cc in corner_cases)
    ax5.bar(list(cc_types.keys()), list(cc_types.values()),
            color='coral', alpha=0.8)
ax5.set_title('Corner Cases Detected', fontsize=11)
ax5.set_ylabel('Count')
ax5.grid(True, alpha=0.3, axis='y')

ax6 = fig.add_subplot(gs[2, 2:])
metrics = {
    'FID'    : fid_fvd['fid']    if fid_fvd else 0,
    'FVD'    : fid_fvd['fvd']    if fid_fvd else 0,
    'Reward' : np.mean([r['total'] for r in reward_data]) if reward_data else 0,
    'CC Found': len(corner_cases) if corner_cases else 0
}
ax6.bar(list(metrics.keys()), list(metrics.values()),
        color=['steelblue', 'coral', 'green', 'orange'],
        alpha=0.8)
ax6.set_title('Key Metrics Summary', fontsize=11)
ax6.set_ylabel('Value')
ax6.grid(True, alpha=0.3, axis='y')

out_path = os.path.join(OUTPUT_DIR, 'pipeline_summary.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Summary figure saved: {out_path}")
plt.show()
plt.close()