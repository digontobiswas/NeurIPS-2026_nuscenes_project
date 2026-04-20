import os
import networkx as nx
import matplotlib.pyplot as plt

print("CausalCoop-WM Plot Causal Graph")
print("==================================================")

graph_path = "outputs/causal_graphs/causal_graph.gpickle"
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

if os.path.exists(graph_path):
    G = nx.read_gpickle(graph_path)
    
    # Subsample for clear plot
    subgraph_nodes = list(G.nodes())[:20]
    G_sub = G.subgraph(subgraph_nodes)
    
    pos = nx.spring_layout(G_sub, seed=42)
    
    plt.figure(figsize=(12, 9))
    nx.draw(G_sub, pos, 
            node_color="lightblue",
            node_size=700,
            with_labels=True,
            font_size=9,
            arrows=True,
            arrowstyle="->",
            arrowsize=12,
            edge_color="gray")
    
    plt.title("Causal Graph (first 20 agents)")
    out_path = os.path.join(output_dir, "causal_graph_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Causal graph plot saved to " + out_path)
else:
    print("Causal graph not found. Run causal model scripts first.")

print("Causal graph plotting completed.")