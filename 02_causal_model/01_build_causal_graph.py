import os
import pickle
import numpy as np
import networkx as nx

TRAJECTORY_DIR = 'outputs/trajectories'
OUTPUT_DIR     = 'outputs/causal_graphs'
GRAPH_PATH     = os.path.join(OUTPUT_DIR, 'causal_graph.gpickle')
MAX_DISTANCE   = 20.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

traj_files = [f for f in os.listdir(TRAJECTORY_DIR) if f.endswith('.pkl')]

if len(traj_files) == 0:
    print("No trajectory files found. Run 03_extract_trajectories.py first.")
    exit()

with open(os.path.join(TRAJECTORY_DIR, traj_files[0]), 'rb') as f:
    trajectories = pickle.load(f)

G      = nx.DiGraph()
agents = list(trajectories.keys())

for inst_token, traj in trajectories.items():
    G.add_node(
        inst_token,
        category=traj[0]['category'],
        frames=len(traj)
    )

print(f"Building causal graph for {len(agents)} agents...")

for i in range(len(agents)):
    for j in range(len(agents)):
        if i == j:
            continue

        traj_i  = trajectories[agents[i]]
        traj_j  = trajectories[agents[j]]
        min_len = min(len(traj_i), len(traj_j))

        if min_len == 0:
            continue

        distances = []
        for k in range(min_len):
            dx = traj_i[k]['x'] - traj_j[k]['x']
            dy = traj_i[k]['y'] - traj_j[k]['y']
            distances.append(np.sqrt(dx**2 + dy**2))

        avg_dist = np.mean(distances)
        min_dist = np.min(distances)

        if min_dist < MAX_DISTANCE:
            weight = 1.0 / (min_dist + 1e-6)
            G.add_edge(
                agents[i], agents[j],
                weight=weight,
                avg_distance=float(avg_dist),
                min_distance=float(min_dist)
            )

import pickle as pkl
with open(GRAPH_PATH, 'wb') as f:
    pkl.dump(G, f)

print(f"Causal graph saved : {GRAPH_PATH}")
print(f"  Nodes            : {G.number_of_nodes()}")
print(f"  Edges            : {G.number_of_edges()}")
print()

print(f"{'Node':<12} {'Category':<30} {'In-degree':<12} {'Out-degree'}")
print('-' * 65)
for node in list(G.nodes())[:10]:
    cat     = G.nodes[node].get('category', 'unknown')
    in_deg  = G.in_degree(node)
    out_deg = G.out_degree(node)
    print(f"{node[:10]:<12} {cat:<30} {in_deg:<12} {out_deg}")