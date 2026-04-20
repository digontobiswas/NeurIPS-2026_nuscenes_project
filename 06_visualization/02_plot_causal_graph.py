import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

GRAPH_PATH  = 'outputs/causal_graphs/causal_graph.gpickle'
INTENT_PATH = 'outputs/causal_graphs/agent_intents.pkl'
OUTPUT_DIR  = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(GRAPH_PATH):
    print("Causal graph not found.")
    print("Run 02_causal_model/01_build_causal_graph.py first.")
    exit()

import pickle as pkl
with open(GRAPH_PATH, 'rb') as f:
    G = pkl.load(f)

with open(INTENT_PATH, 'rb') as f:
    intent_data = pickle.load(f)

intent_colors = {
    'moving_straight': '#4C72B0',
    'turning_left'   : '#55A868',
    'turning_right'  : '#C44E52',
    'stopping'       : '#DD8452',
    'stationary'     : '#8172B2',
    'unknown'        : '#CCCCCC'
}

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('Causal Graph Analysis', fontsize=14)

pos          = nx.spring_layout(G, seed=42, k=2.0)
node_colors  = [
    intent_colors.get(
        intent_data.get(n, {}).get('intent', 'unknown'),
        '#CCCCCC'
    )
    for n in G.nodes()
]
edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
max_w        = max(edge_weights) if edge_weights else 1.0

nx.draw_networkx_edges(
    G, pos, ax=axes[0],
    alpha=0.3,
    width=[w / max_w * 3 for w in edge_weights],
    edge_color='gray', arrows=True
)
nx.draw_networkx_nodes(
    G, pos, ax=axes[0],
    node_color=node_colors,
    node_size=300, alpha=0.9
)

for intent, color in intent_colors.items():
    axes[0].plot([], [], 'o', color=color, label=intent)
axes[0].legend(loc='upper left', fontsize=7)
axes[0].set_title('Causal Graph (by intent)')
axes[0].axis('off')

in_degrees  = [G.in_degree(n)  for n in G.nodes()]
out_degrees = [G.out_degree(n) for n in G.nodes()]

axes[1].scatter(in_degrees, out_degrees,
                c=node_colors, s=60, alpha=0.8)
axes[1].set_title('In-degree vs Out-degree')
axes[1].set_xlabel('In-degree')
axes[1].set_ylabel('Out-degree')
axes[1].grid(True, alpha=0.3)

centrality = nx.degree_centrality(G)
cent_vals  = list(centrality.values())
axes[2].hist(cent_vals, bins=15,
             color='steelblue', alpha=0.8, edgecolor='white')
axes[2].set_title('Degree Centrality Distribution')
axes[2].set_xlabel('Centrality')
axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'causal_graph_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()