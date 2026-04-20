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
    print("Causal graph not found. Run 01_build_causal_graph.py first.")
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

node_colors = []
for node in G.nodes():
    intent = intent_data.get(node, {}).get('intent', 'unknown')
    node_colors.append(intent_colors.get(intent, '#CCCCCC'))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Causal Graph — Agent Interaction Network', fontsize=14)

pos = nx.spring_layout(G, seed=42, k=2.5)

edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
max_w        = max(edge_weights) if edge_weights else 1.0
norm_weights = [w / max_w for w in edge_weights]

nx.draw_networkx_edges(
    G, pos, ax=axes[0],
    alpha=0.3,
    width=[w * 2.5 for w in norm_weights],
    edge_color='gray',
    arrows=True,
    arrowsize=12
)
nx.draw_networkx_nodes(
    G, pos, ax=axes[0],
    node_color=node_colors,
    node_size=400,
    alpha=0.9
)

labels = {
    n: intent_data.get(n, {}).get('intent', '?')[:6]
    for n in G.nodes()
}
nx.draw_networkx_labels(G, pos, labels, ax=axes[0], font_size=6)

for intent, color in intent_colors.items():
    axes[0].plot([], [], 'o', color=color, label=intent)
axes[0].legend(loc='upper left', fontsize=8)
axes[0].set_title('Agent Causal Graph (colored by intent)')
axes[0].axis('off')

degrees    = dict(G.degree())
deg_values = list(degrees.values())
axes[1].hist(deg_values, bins=15, color='steelblue',
             alpha=0.8, edgecolor='white')
axes[1].set_title('Node Degree Distribution')
axes[1].set_xlabel('Degree (connections)')
axes[1].set_ylabel('Number of agents')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'causal_graph.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()