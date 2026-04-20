import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/causal_graphs'
GRAPH_PATH     = os.path.join(OUTPUT_DIR, 'causal_graph.gpickle')
MOVING_GRAPH_PATH = os.path.join(OUTPUT_DIR, 'causal_graph_moving.gpickle')
MAX_DISTANCE   = 20.0
MOVING_THRESH  = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

traj_files = [f for f in os.listdir(TRAJECTORY_DIR) if f.endswith('.pkl')]

if len(traj_files) == 0:
    print("No trajectory files found.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

# ── Separate moving vs stationary agents ──────────────────────
moving_agents     = {}
stationary_agents = {}

for inst_token, traj in trajectories.items():
    if len(traj) < 2:
        stationary_agents[inst_token] = traj
        continue
    dx   = traj[-1]['x'] - traj[0]['x']
    dy   = traj[-1]['y'] - traj[0]['y']
    dist = np.sqrt(dx**2 + dy**2)
    if dist > MOVING_THRESH:
        moving_agents[inst_token] = traj
    else:
        stationary_agents[inst_token] = traj

print(f"Total agents    : {len(trajectories)}")
print(f"Moving agents   : {len(moving_agents)}")
print(f"Stationary      : {len(stationary_agents)}")
print()


def compute_causal_weight(traj_i, traj_j):
    """
    Compute causal influence weight from agent i to agent j.
    Based on:
    - Spatial proximity (closer = stronger influence)
    - Temporal precedence (does i move before j reacts?)
    - Velocity correlation
    Returns weight in [0, 1].
    """
    min_len = min(len(traj_i), len(traj_j))
    if min_len < 2:
        return 0.0

    # Spatial proximity score
    distances = []
    for k in range(min_len):
        dx = traj_i[k]['x'] - traj_j[k]['x']
        dy = traj_i[k]['y'] - traj_j[k]['y']
        distances.append(np.sqrt(dx**2 + dy**2))

    min_dist = np.min(distances)
    if min_dist > MAX_DISTANCE:
        return 0.0

    proximity_score = 1.0 / (1.0 + min_dist)

    # Velocity correlation score
    vel_i = []
    vel_j = []
    for k in range(1, min_len):
        vel_i.append([
            traj_i[k]['x'] - traj_i[k-1]['x'],
            traj_i[k]['y'] - traj_i[k-1]['y']
        ])
        vel_j.append([
            traj_j[k]['x'] - traj_j[k-1]['x'],
            traj_j[k]['y'] - traj_j[k-1]['y']
        ])

    if len(vel_i) == 0:
        return proximity_score

    vel_i  = np.array(vel_i)
    vel_j  = np.array(vel_j)
    speeds_i = np.sqrt(np.sum(vel_i**2, axis=1))
    speeds_j = np.sqrt(np.sum(vel_j**2, axis=1))

    if np.std(speeds_i) > 0 and np.std(speeds_j) > 0:
        corr = np.corrcoef(speeds_i, speeds_j)[0, 1]
        corr_score = (corr + 1) / 2
    else:
        corr_score = 0.5

    weight = 0.6 * proximity_score + 0.4 * corr_score
    return float(np.clip(weight, 0, 1))


# ── Build full graph (all agents) ─────────────────────────────
print("Building full causal graph...")
G_full  = nx.DiGraph()
agents  = list(trajectories.keys())

for inst_token, traj in trajectories.items():
    G_full.add_node(
        inst_token,
        category=traj[0]['category'],
        frames=len(traj),
        is_moving=(inst_token in moving_agents)
    )

for i in range(len(agents)):
    for j in range(len(agents)):
        if i == j:
            continue
        weight = compute_causal_weight(
            trajectories[agents[i]],
            trajectories[agents[j]]
        )
        if weight > 0.05:
            G_full.add_edge(
                agents[i], agents[j],
                weight=weight
            )

with open(GRAPH_PATH, 'wb') as f:
    pickle.dump(G_full, f)

print(f"Full graph     : {G_full.number_of_nodes()} nodes, "
      f"{G_full.number_of_edges()} edges")


# ── Build moving-only graph ───────────────────────────────────
print("Building moving-agent causal graph...")
G_moving = nx.DiGraph()
m_agents = list(moving_agents.keys())

for inst_token, traj in moving_agents.items():
    G_moving.add_node(
        inst_token,
        category=traj[0]['category'],
        frames=len(traj)
    )

for i in range(len(m_agents)):
    for j in range(len(m_agents)):
        if i == j:
            continue
        weight = compute_causal_weight(
            moving_agents[m_agents[i]],
            moving_agents[m_agents[j]]
        )
        if weight > 0.05:
            G_moving.add_edge(
                m_agents[i], m_agents[j],
                weight=weight
            )

with open(MOVING_GRAPH_PATH, 'wb') as f:
    pickle.dump(G_moving, f)

print(f"Moving graph   : {G_moving.number_of_nodes()} nodes, "
      f"{G_moving.number_of_edges()} edges")
print()

# ── Find most influential agents ──────────────────────────────
print("TOP 10 MOST INFLUENTIAL AGENTS (by out-degree):")
print('-' * 60)
print(f"{'#':<5} {'Category':<25} {'Out-deg':<10} {'In-deg':<10} {'Centrality'}")
print('-' * 60)

centrality = nx.degree_centrality(G_moving)
sorted_nodes = sorted(
    G_moving.nodes(),
    key=lambda n: G_moving.out_degree(n),
    reverse=True
)[:10]

for i, node in enumerate(sorted_nodes):
    cat    = G_moving.nodes[node].get('category', '?').split('.')[-1]
    out_d  = G_moving.out_degree(node)
    in_d   = G_moving.in_degree(node)
    cent   = centrality.get(node, 0)
    print(f"{i+1:<5} {cat:<25} {out_d:<10} {in_d:<10} {cent:.4f}")

# ── Visualize moving agent graph ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle(
    'Causal Graph — Moving Agents Only\n'
    'Edge weight = proximity × velocity correlation',
    fontsize=13
)

pos = nx.spring_layout(G_moving, seed=42, k=3.0)

cat_colors = {
    'car'        : '#4C72B0',
    'pedestrian' : '#DD8452',
    'truck'      : '#55A868',
    'bus'        : '#C44E52',
    'motorcycle' : '#8172B2',
    'bicycle'    : '#937860',
    'trailer'    : '#DA8EC0'
}

node_colors = []
for node in G_moving.nodes():
    cat   = G_moving.nodes[node].get('category', '').split('.')[-1]
    color = cat_colors.get(cat, '#CCCCCC')
    node_colors.append(color)

edge_weights = [G_moving[u][v]['weight'] for u, v in G_moving.edges()]
max_w        = max(edge_weights) if edge_weights else 1.0

nx.draw_networkx_edges(
    G_moving, pos, ax=axes[0],
    alpha=0.4,
    width=[w / max_w * 3 for w in edge_weights],
    edge_color='gray',
    arrows=True,
    arrowsize=10
)
nx.draw_networkx_nodes(
    G_moving, pos, ax=axes[0],
    node_color=node_colors,
    node_size=500,
    alpha=0.9
)
labels = {
    n: G_moving.nodes[n].get('category', '?').split('.')[-1][:4]
    for n in G_moving.nodes()
}
nx.draw_networkx_labels(
    G_moving, pos, labels,
    ax=axes[0], font_size=7
)

for cat, color in cat_colors.items():
    axes[0].plot([], [], 'o', color=color, label=cat)
axes[0].legend(loc='upper left', fontsize=8)
axes[0].set_title(f'Causal Graph ({G_moving.number_of_nodes()} moving agents)')
axes[0].axis('off')

# Edge weight distribution
axes[1].hist(
    edge_weights, bins=30,
    color='steelblue', alpha=0.8,
    edgecolor='white'
)
axes[1].set_title('Causal Edge Weight Distribution')
axes[1].set_xlabel('Weight (proximity × velocity correlation)')
axes[1].set_ylabel('Number of edges')
axes[1].axvline(
    x=np.mean(edge_weights),
    color='red', linestyle='--',
    label=f'Mean={np.mean(edge_weights):.3f}'
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

out_path = 'outputs/figures/causal_graph_moving.png'
os.makedirs('outputs/figures', exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out_path}")
plt.show()
plt.close()

print()
print("SUMMARY:")
print(f"  Full graph    : {G_full.number_of_nodes()} nodes, "
      f"{G_full.number_of_edges()} edges")
print(f"  Moving graph  : {G_moving.number_of_nodes()} nodes, "
      f"{G_moving.number_of_edges()} edges")
print(f"  Edge weight   : proximity × velocity correlation")
print(f"  Ready for     : counterfactual queries + intent inference")