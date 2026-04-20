import os
import torch
import numpy as np
import pickle

print("CausalCoop-WM Dynamic Prior")
print("==================================================")

trajectory_dir = "outputs/trajectories"
output_dir = "outputs/world_model_data"
os.makedirs(output_dir, exist_ok=True)

# Load trajectories
trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
dynamic_prior = {}

for tf in trajectory_files[:10]:  # limit to first 10 agents for speed
    with open(os.path.join(trajectory_dir, tf), "rb") as f:
        traj = pickle.load(f)
    instance_token = tf.replace("traj_", "").replace(".pkl", "")
    
    # Compute velocity and acceleration prior
    positions = np.array([p["translation"][:2] for p in traj])
    velocities = np.diff(positions, axis=0)
    if len(velocities) > 0:
        mean_vel = np.mean(velocities, axis=0)
        std_vel = np.std(velocities, axis=0)
    else:
        mean_vel = np.zeros(2)
        std_vel = np.ones(2)
    
    dynamic_prior[instance_token] = {
        "mean_velocity": mean_vel.tolist(),
        "std_velocity": std_vel.tolist(),
        "category": traj[0]["category"]
    }

torch.save(dynamic_prior, os.path.join(output_dir, "dynamic_prior.pt"))
print("Dynamic prior computed for " + str(len(dynamic_prior)) + " agents")
print("Dynamic prior saved to outputs/world_model_data/dynamic_prior.pt")
print("Dynamic prior generation completed.")