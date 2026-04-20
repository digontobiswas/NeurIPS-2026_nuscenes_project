import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import torch

def plot_trajectories(trajectory_dir, output_path=None):
    print("Plotting trajectories")
    trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
    
    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(trajectory_files))))
    
    for idx, tf in enumerate(trajectory_files[:10]):
        with open(os.path.join(trajectory_dir, tf), "rb") as f:
            traj = pickle.load(f)
        positions = np.array([p["translation"][:2] for p in traj])
        plt.plot(positions[:, 0], positions[:, 1], color=colors[idx], linewidth=2, label="Agent " + tf.replace("traj_", "").replace(".pkl", "")[:8])
    
    plt.title("Agent Trajectories (Top-Down View)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("Trajectory plot saved to " + output_path)
    plt.show()

def plot_causal_graph(graph_path, output_path=None):
    print("Plotting causal graph")
    if not os.path.exists(graph_path):
        print("Graph not found")
        return
    G = nx.read_gpickle(graph_path) if hasattr(nx, 'read_gpickle') else nx.read_gpickle(graph_path)
    subgraph_nodes = list(G.nodes())[:20]
    G_sub = G.subgraph(subgraph_nodes)
    pos = nx.spring_layout(G_sub, seed=42)
    
    plt.figure(figsize=(12, 9))
    nx.draw(G_sub, pos, node_color="lightblue", node_size=700, with_labels=True, font_size=9, arrows=True, arrowstyle="->", arrowsize=12, edge_color="gray")
    plt.title("Causal Graph (first 20 agents)")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("Causal graph plot saved to " + output_path)
    plt.show()

def plot_attention_map(fused_belief_path, output_path=None):
    print("Plotting attention map")
    if not os.path.exists(fused_belief_path):
        print("Fused belief not found")
        return
    fused_belief = torch.load(fused_belief_path)
    
    # Fixed line with type ignore to remove yellow underline
    attention_map = torch.softmax(fused_belief[:64].unsqueeze(0), dim=1).squeeze(0).reshape(8, 8).cpu().numpy()  # type: ignore[attr-defined]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map, cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.title("Simulated Attention Map (Belief Fusion)")
    plt.xlabel("Latent Dimension X")
    plt.ylabel("Latent Dimension Y")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print("Attention map saved to " + output_path)
    plt.show()

def save_figure(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Figure saved to " + output_path)