import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx


def plot_trajectories(trajectory_dir, output_dir='outputs/figures'):
    """
    Plot agent trajectories from saved pkl files.
    """
    os.makedirs(output_dir, exist_ok=True)

    traj_files = [
        f for f in os.listdir(trajectory_dir)
        if f.endswith('.pkl')
    ]

    if len(traj_files) == 0:
        print("No trajectory files found.")
        return

    traj_path = os.path.join(trajectory_dir, traj_files[0])
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors  = cm.tab20(np.linspace(0, 1, len(trajectories)))

    for (inst_token, traj), color in zip(trajectories.items(), colors):
        xs = [p['x'] for p in traj]
        ys = [p['y'] for p in traj]

        ax.plot(xs, ys, '-o', color=color,
                markersize=3, linewidth=1.5, alpha=0.8)
        ax.plot(xs[0], ys[0], 's', color=color,
                markersize=6, label=traj[0]['category'].split('.')[1])
        ax.plot(xs[-1], ys[-1], '^', color=color, markersize=6)

    ax.set_title('Agent Trajectories — nuScenes Scene 0', fontsize=13)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(),
              loc='upper right', fontsize=8)

    out_path = os.path.join(output_dir, 'trajectories.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Trajectory plot saved: {out_path}")
    plt.show()
    plt.close()


def plot_causal_graph(graph_path, output_dir='outputs/figures'):
    """
    Visualize the causal graph with nodes colored by category.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(graph_path):
        print(f"Graph not found: {graph_path}")
        return

    G = nx.read_gpickle(graph_path)

    if G.number_of_nodes() == 0:
        print("Graph is empty.")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    # Color nodes by category
    category_colors = {
        'vehicle'   : '#4C72B0',
        'human'     : '#DD8452',
        'movable'   : '#55A868',
        'static'    : '#C44E52',
        'unknown'   : '#8172B2'
    }

    node_colors = []
    for node in G.nodes():
        cat = G.nodes[node].get('category', 'unknown')
        top = cat.split('.')[0] if '.' in cat else 'unknown'
        color = category_colors.get(top, '#8172B2')
        node_colors.append(color)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Draw edges
    edge_weights = [
        G[u][v].get('weight', 1.0)
        for u, v in G.edges()
    ]
    max_w = max(edge_weights) if edge_weights else 1.0
    norm_weights = [w / max_w for w in edge_weights]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        alpha=0.4,
        width=[w * 2 for w in norm_weights],
        edge_color='gray',
        arrows=True,
        arrowsize=15
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=300,
        alpha=0.9
    )

    # Short labels
    labels = {
        n: G.nodes[n].get('category', '?').split('.')[-1][:8]
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    # Legend
    for cat, color in category_colors.items():
        ax.plot([], [], 'o', color=color, label=cat)
    ax.legend(loc='upper left', fontsize=9)

    ax.set_title('Causal Graph — Agent Interactions', fontsize=13)
    ax.axis('off')

    out_path = os.path.join(output_dir, 'causal_graph.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Causal graph saved: {out_path}")
    plt.show()
    plt.close()


def plot_attention_map(fused_belief_path, output_dir='outputs/figures'):
    """
    Visualize the V2X fused belief as an attention heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(fused_belief_path):
        print(f"Fused belief not found: {fused_belief_path}")
        return

    import torch
    belief = torch.load(fused_belief_path, weights_only=False)

    if hasattr(belief, 'numpy'):
        data = belief.numpy()
    else:
        data = np.array(belief)

    # Reshape for heatmap if needed
    if data.ndim == 1:
        size = int(np.ceil(np.sqrt(len(data))))
        pad  = size * size - len(data)
        data = np.pad(data, (0, pad))
        data = data.reshape(size, size)
    elif data.ndim > 2:
        data = data.reshape(data.shape[0], -1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('V2X Fused Belief — Latent Attention Map', fontsize=13)

    # Heatmap
    im = axes[0].imshow(data, cmap='hot', aspect='auto')
    axes[0].set_title('Belief Heatmap')
    axes[0].set_xlabel('Latent dimension')
    axes[0].set_ylabel('Agent / time')
    plt.colorbar(im, ax=axes[0])

    # Distribution
    axes[1].hist(data.flatten(), bins=50,
                 color='steelblue', alpha=0.8, edgecolor='white')
    axes[1].set_title('Belief Value Distribution')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')

    out_path = os.path.join(output_dir, 'attention_map.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Attention map saved: {out_path}")
    plt.show()
    plt.close()