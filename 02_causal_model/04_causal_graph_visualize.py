import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

print("CausalCoop-WM Causal Graph Visualization")
print("==================================================")

graph_path = "outputs/causal_graphs/causal_graph.gpickle"
intent_path = "outputs/causal_graphs/agent_intents.pkl"
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(graph_path) or not os.path.exists(intent_path):
    print("Causal graph or intents not found. Run previous scripts first.")
else:
    G = nx.read_gpickle(graph_path)
    with open(intent_path, "rb") as f:
        intent_data = pickle.load(f)
    
    # Subsample for clear visualization
    subgraph_nodes = list(G.nodes())[:15]
    G_sub = G.subgraph(subgraph_nodes)
    
    pos = nx.spring_layout(G_sub, seed=42)
    
    plt.figure(figsize=(12, 8))
    node_colors = []
    for node in G_sub.nodes():
        intent = intent_data.get(node, {}).get("most_common_intent", "unknown")
        color_map = {"braking": "red", "accelerating": "green", "turning": "blue", "cruising": "gray"}
        node_colors.append(color_map.get(intent, "gray"))
    
    nx.draw(G_sub, pos, 
            node_color=node_colors,
            node_size=800,
            with_labels=True,
            font_size=8,
            arrows=True,
            arrowstyle="->",
            arrowsize=15)
    
    plt.title("Causal Graph Visualization (first 15 agents)")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "causal_graph_visualization.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Causal graph visualization saved to " + out_path)
    print("Visualization completed.")