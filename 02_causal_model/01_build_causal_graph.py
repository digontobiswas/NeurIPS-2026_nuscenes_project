import os
import pickle
import networkx as nx
import numpy as np

print("CausalCoop-WM Build Causal Graph")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
trajectory_dir = "outputs/trajectories"
output_dir = "outputs/causal_graphs"
os.makedirs(output_dir, exist_ok=True)

# Load all trajectories
trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
all_trajectories = {}

for tf in trajectory_files:
    with open(os.path.join(trajectory_dir, tf), "rb") as f:
        traj = pickle.load(f)
    instance_token = tf.replace("traj_", "").replace(".pkl", "")
    all_trajectories[instance_token] = traj

print("Loaded " + str(len(all_trajectories)) + " agent trajectories")

# Build causal graph
G = nx.DiGraph()

for instance_token, traj in all_trajectories.items():
    G.add_node(instance_token, 
               category=traj[0]["category"], 
               frames=len(traj))

# Add directed edges based on potential causal influence (proximity + velocity)
for i1, traj1 in all_trajectories.items():
    for i2, traj2 in all_trajectories.items():
        if i1 == i2:
            continue
        for t in range(min(len(traj1), len(traj2))):
            pos1 = np.array(traj1[t]["translation"][:2])
            pos2 = np.array(traj2[t]["translation"][:2])
            dist = np.linalg.norm(pos1 - pos2)
            if dist < 15.0:  # within 15m causal influence range
                vel1 = np.array(traj1[t].get("velocity", [0,0])) if "velocity" in traj1[t] else np.zeros(2)
                vel2 = np.array(traj2[t].get("velocity", [0,0])) if "velocity" in traj2[t] else np.zeros(2)
                if np.dot(vel1, vel2) < 0:  # opposing or following motion
                    G.add_edge(i1, i2, weight=1.0 / (dist + 1e-6), time=t)
                    break

nx.write_gpickle(G, os.path.join(output_dir, "causal_graph.gpickle"))
print("Causal graph built with " + str(G.number_of_nodes()) + " nodes and " + str(G.number_of_edges()) + " edges")
print("Graph saved to outputs/causal_graphs/causal_graph.gpickle")
print("Causal graph construction completed.")