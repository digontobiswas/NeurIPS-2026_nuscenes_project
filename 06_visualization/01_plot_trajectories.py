import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

print("CausalCoop-WM Plot Trajectories")
print("==================================================")

trajectory_dir = "outputs/trajectories"
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

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
out_path = os.path.join(output_dir, "trajectories_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Trajectories plot saved to " + out_path)
print("Trajectory plotting completed.")