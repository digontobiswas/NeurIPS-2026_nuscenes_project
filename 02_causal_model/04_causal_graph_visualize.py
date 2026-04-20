import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

TRAJECTORY_DIR    = 'outputs/trajectories'
GRAPH_PATH        = 'outputs/causal_graphs/causal_graph_moving.gpickle'
OUTPUT_DIR        = 'outputs/causal_graphs'
INTENT_PATH       = os.path.join(OUTPUT_DIR, 'agent_intents.pkl')
FIGURES           = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES,    exist_ok=True)

# ── Load trajectories ─────────────────────────────────────────
traj_files = [f for f in os.listdir(TRAJECTORY_DIR) if f.endswith('.pkl')]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

# ── Load moving agent graph ───────────────────────────────────
if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH, 'rb') as f:
        G_moving = pickle.load(f)
    moving_tokens = set(G_moving.nodes())
    print(f"Moving agents loaded from graph: {len(moving_tokens)}")
else:
    print("Moving graph not found. Using distance threshold.")
    moving_tokens = set()
    for token, traj in trajectories.items():
        if len(traj) > 1:
            dx   = traj[-1]['x'] - traj[0]['x']
            dy   = traj[-1]['y'] - traj[0]['y']
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1.0:
                moving_tokens.add(token)


def compute_detailed_intent(traj):
    """
    Compute detailed intent from trajectory using:
    - Total displacement (direction)
    - Speed profile (acceleration pattern)
    - Heading change (turning radius)
    Returns intent label + confidence score.
    """
    if len(traj) < 2:
        return 'stationary', 1.0, {}

    positions = np.array([[p['x'], p['y']] for p in traj])
    times     = np.array([p['timestamp'] for p in traj]) / 1e6

    # Total displacement
    total_dx   = positions[-1, 0] - positions[0, 0]
    total_dy   = positions[-1, 1] - positions[0, 1]
    total_dist = np.sqrt(total_dx**2 + total_dy**2)

    if total_dist < 1.0:
        return 'stationary', 0.95, {
            'displacement': total_dist,
            'avg_speed'   : 0.0
        }

    # Speed profile
    diffs  = np.diff(positions, axis=0)
    dt     = np.diff(times)
    dt     = np.where(dt > 0, dt, 1e-6)
    speeds = np.sqrt(np.sum(diffs**2, axis=1)) / dt

    avg_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))

    # Heading change — measure turning
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    if len(headings) > 1:
        heading_changes = np.abs(np.diff(headings))
        heading_changes = np.where(
            heading_changes > np.pi,
            2 * np.pi - heading_changes,
            heading_changes
        )
        total_heading_change = float(np.sum(heading_changes))
    else:
        total_heading_change = 0.0

    # Classify based on heading change + displacement direction
    if avg_speed < 0.3:
        intent     = 'stopping'
        confidence = 0.85
    elif total_heading_change > 0.8:
        if total_dy > 0:
            intent     = 'turning_left'
            confidence = min(0.95, total_heading_change)
        else:
            intent     = 'turning_right'
            confidence = min(0.95, total_heading_change)
    elif abs(total_dy) < 3.0 and total_dist > 2.0:
        intent     = 'moving_straight'
        confidence = 0.9
    elif avg_speed > 5.0:
        intent     = 'moving_fast'
        confidence = 0.8
    else:
        intent     = 'moving_straight'
        confidence = 0.7

    return intent, confidence, {
        'displacement'        : float(total_dist),
        'avg_speed'           : avg_speed,
        'max_speed'           : max_speed,
        'total_heading_change': total_heading_change
    }


# ── Compute intents for all agents ────────────────────────────
intent_data    = {}
moving_intents = {}

print()
print("Computing agent intents...")
print()

for inst_token, traj in trajectories.items():
    intent, confidence, stats = compute_detailed_intent(traj)

    intent_data[inst_token] = {
        'intent'    : intent,
        'confidence': confidence,
        'category'  : traj[0]['category'],
        'frames'    : len(traj),
        'stats'     : stats,
        'is_moving' : inst_token in moving_tokens
    }

    if inst_token in moving_tokens:
        moving_intents[inst_token] = intent_data[inst_token]

# ── Save ──────────────────────────────────────────────────────
with open(INTENT_PATH, 'wb') as f:
    pickle.dump(intent_data, f)

print(f"Intent data saved: {INTENT_PATH}")
print(f"Total agents     : {len(intent_data)}")
print(f"Moving agents    : {len(moving_intents)}")
print()

# ── Print moving agent intent table ───────────────────────────
print("MOVING AGENT INTENT ANALYSIS:")
print('=' * 75)
print(f"{'#':<5} {'Category':<25} {'Intent':<20} {'Conf':<8} "
      f"{'Dist(m)':<10} {'Speed(m/s)'}")
print('-' * 75)

for i, (token, data) in enumerate(moving_intents.items()):
    cat   = data['category'].split('.')[-1]
    stats = data['stats']
    dist  = stats.get('displacement', 0)
    speed = stats.get('avg_speed', 0)
    print(
        f"{i+1:<5} {cat:<25} {data['intent']:<20} "
        f"{data['confidence']:<8.2f} {dist:<10.2f} {speed:.2f}"
    )

# ── Intent distribution for moving agents ─────────────────────
print()
print("INTENT DISTRIBUTION (moving agents only):")
print('-' * 40)
m_intents = [d['intent'] for d in moving_intents.values()]
counts    = Counter(m_intents)
for intent, count in sorted(counts.items(), key=lambda x: -x[1]):
    bar = '█' * count
    print(f"  {intent:<20} {count:>3}  {bar}")

# ── Visualize ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    'Agent Intent Analysis — PS-1 Causal Scene Understanding',
    fontsize=13
)

# Plot 1 — Intent distribution pie chart
intent_labels  = list(counts.keys())
intent_values  = list(counts.values())
intent_colors  = {
    'moving_straight': '#4C72B0',
    'turning_left'   : '#55A868',
    'turning_right'  : '#C44E52',
    'stopping'       : '#DD8452',
    'stationary'     : '#8172B2',
    'moving_fast'    : '#E377C2'
}
pie_colors = [intent_colors.get(l, '#CCCCCC') for l in intent_labels]

axes[0].pie(
    intent_values,
    labels=intent_labels,
    colors=pie_colors,
    autopct='%1.1f%%',
    startangle=90
)
axes[0].set_title('Intent Distribution\n(moving agents)')

# Plot 2 — Speed vs displacement scatter
speeds = [
    d['stats'].get('avg_speed', 0)
    for d in moving_intents.values()
]
displacements = [
    d['stats'].get('displacement', 0)
    for d in moving_intents.values()
]
colors_scatter = [
    intent_colors.get(d['intent'], '#CCCCCC')
    for d in moving_intents.values()
]

axes[1].scatter(
    displacements, speeds,
    c=colors_scatter, s=80, alpha=0.8
)
for intent, color in intent_colors.items():
    axes[1].scatter([], [], c=color, label=intent, s=60)
axes[1].legend(fontsize=8, loc='upper left')
axes[1].set_title('Speed vs Displacement\nby Intent')
axes[1].set_xlabel('Total displacement (m)')
axes[1].set_ylabel('Average speed (m/s)')
axes[1].grid(True, alpha=0.3)

# Plot 3 — Confidence distribution
confidences = [d['confidence'] for d in moving_intents.values()]
axes[2].hist(
    confidences, bins=15,
    color='steelblue', alpha=0.8,
    edgecolor='white'
)
axes[2].axvline(
    x=np.mean(confidences),
    color='red', linestyle='--',
    label=f'Mean={np.mean(confidences):.2f}'
)
axes[2].set_title('Intent Confidence Distribution')
axes[2].set_xlabel('Confidence score')
axes[2].set_ylabel('Number of agents')
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(FIGURES, 'intent_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nIntent analysis plot saved: {out_path}")
plt.show()
plt.close()

print()
print("READY FOR: counterfactual queries using inferred intents")