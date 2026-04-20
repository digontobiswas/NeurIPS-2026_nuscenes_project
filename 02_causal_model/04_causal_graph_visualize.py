import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx

GRAPH_PATH  = 'outputs/causal_graphs/causal_graph_moving.gpickle'
INTENT_PATH = 'outputs/causal_graphs/agent_intents.pkl'
TRAJ_DIR    = 'outputs/trajectories'
OUTPUT_DIR  = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(GRAPH_PATH):
    print("Moving graph not found.")
    exit()

with open(GRAPH_PATH, 'rb') as f:
    G = pickle.load(f)

with open(INTENT_PATH, 'rb') as f:
    intent_data = pickle.load(f)

traj_files = [f for f in os.listdir(TRAJ_DIR) if f.endswith('.pkl')]
with open(os.path.join(TRAJ_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

intent_colors = {
    'moving_straight': '#4C72B0',
    'turning_left'   : '#55A868',
    'turning_right'  : '#C44E52',
    'stopping'       : '#DD8452',
    'stationary'     : '#8172B2',
    'moving_fast'    : '#E377C2',
    'unknown'        : '#CCCCCC'
}

fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    'CausalCoop-WM — Causal Scene Understanding\n'
    'PS-1: Structural Causal Graph over Agent Intent',
    fontsize=14, fontweight='bold'
)

# ── Plot 1 — Causal graph colored by intent ───────────────────
ax1 = fig.add_subplot(2, 2, 1)

pos = nx.spring_layout(G, seed=42, k=3.0)

node_colors = [
    intent_colors.get(
        intent_data.get(n, {}).get('intent', 'unknown'),
        '#CCCCCC'
    )
    for n in G.nodes()
]

node_sizes = [
    300 + G.out_degree(n) * 30
    for n in G.nodes()
]

edge_weights = [G[u][v].get('weight', 0.1) for u, v in G.edges()]
max_w        = max(edge_weights) if edge_weights else 1.0

nx.draw_networkx_edges(
    G, pos, ax=ax1,
    alpha=0.3,
    width=[w / max_w * 2.5 for w in edge_weights],
    edge_color='gray',
    arrows=True,
    arrowsize=8
)
nx.draw_networkx_nodes(
    G, pos, ax=ax1,
    node_color=node_colors,
    node_size=node_sizes,
    alpha=0.9
)

for intent, color in intent_colors.items():
    ax1.plot([], [], 'o', color=color, label=intent, markersize=8)
ax1.legend(loc='upper left', fontsize=7, ncol=2)
ax1.set_title('Causal Graph\n(node size = out-degree)', fontsize=10)
ax1.axis('off')

# ── Plot 2 — Trajectory map with intent colors ────────────────
ax2 = fig.add_subplot(2, 2, 2)

for inst_token, traj in trajectories.items():
    if not intent_data.get(inst_token, {}).get('is_moving', False):
        continue

    xs     = [p['x'] for p in traj]
    ys     = [p['y'] for p in traj]
    intent = intent_data.get(inst_token, {}).get('intent', 'unknown')
    color  = intent_colors.get(intent, '#CCCCCC')

    ax2.plot(xs, ys, '-', color=color, linewidth=2, alpha=0.8)
    ax2.plot(xs[0],  ys[0],  's', color=color, markersize=6)
    ax2.plot(xs[-1], ys[-1], '^', color=color, markersize=6)

for intent, color in intent_colors.items():
    ax2.plot([], [], '-', color=color, label=intent, linewidth=2)
ax2.legend(loc='upper right', fontsize=7)
ax2.set_title('Agent Trajectories\nColored by Intent', fontsize=10)
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Y (m)')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.2)

# ── Plot 3 — Degree centrality distribution ───────────────────
ax3 = fig.add_subplot(2, 2, 3)

centrality   = nx.degree_centrality(G)
betweenness  = nx.betweenness_centrality(G)

cent_values  = list(centrality.values())
bet_values   = list(betweenness.values())

node_intent_colors = [
    intent_colors.get(
        intent_data.get(n, {}).get('intent', 'unknown'),
        '#CCCCCC'
    )
    for n in G.nodes()
]

ax3.scatter(cent_values, bet_values,
            c=node_intent_colors, s=60, alpha=0.8)

for intent, color in intent_colors.items():
    ax3.scatter([], [], c=color, label=intent, s=40)
ax3.legend(fontsize=7, loc='upper left')
ax3.set_title('Degree vs Betweenness Centrality\n(by intent)', fontsize=10)
ax3.set_xlabel('Degree centrality')
ax3.set_ylabel('Betweenness centrality')
ax3.grid(True, alpha=0.3)

# ── Plot 4 — Edge weight vs distance heatmap ──────────────────
ax4 = fig.add_subplot(2, 2, 4)

weights   = []
min_dists = []

for u, v, data in G.edges(data=True):
    weights.append(data.get('weight', 0))
    min_dists.append(data.get('min_distance', 10))

if weights:
    ax4.scatter(min_dists, weights,
                alpha=0.4, s=20, color='steelblue')
    z = np.polyfit(min_dists, weights, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(min_dists), max(min_dists), 100)
    ax4.plot(x_line, p(x_line), 'r--',
             linewidth=2, label='Trend')
    ax4.legend()

ax4.set_title('Causal Weight vs Agent Distance\n'
              '(confirms proximity drives causality)',
              fontsize=10)
ax4.set_xlabel('Minimum distance between agents (m)')
ax4.set_ylabel('Causal edge weight')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'causal_full_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()