import os
import torch
import numpy as np
import pickle

print("CausalCoop-WM Trajectory Difference")
print("==================================================")

trajectory_dir = "outputs/trajectories"
world_model_dir = "outputs/world_model_data"
output_dir = "outputs/evaluation"
os.makedirs(output_dir, exist_ok=True)

future_path = os.path.join(world_model_dir, "future_prediction.pt")
if os.path.exists(future_path):
    predicted_positions = torch.load(future_path)
    print("Loaded predicted future positions with " + str(len(predicted_positions)) + " frames")
    
    # Load one ground truth trajectory for comparison
    trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
    if trajectory_files:
        with open(os.path.join(trajectory_dir, trajectory_files[0]), "rb") as f:
            gt_traj = pickle.load(f)
        
        gt_positions = np.array([p["translation"][:2] for p in gt_traj[-len(predicted_positions):]])
        pred_positions = np.array([p[:2] for p in predicted_positions])
        
        # Compute difference
        diff = np.mean(np.linalg.norm(gt_positions - pred_positions, axis=1))
        rmse = np.sqrt(np.mean((gt_positions - pred_positions)**2))
        
        results = {
            "mean_position_difference": float(diff),
            "rmse": float(rmse),
            "num_frames_compared": len(predicted_positions)
        }
        
        torch.save(results, os.path.join(output_dir, "trajectory_difference.pt"))
        print("Mean position difference: " + str(round(diff, 4)) + " meters")
        print("RMSE: " + str(round(rmse, 4)) + " meters")
        print("Trajectory difference saved to outputs/evaluation/trajectory_difference.pt")
    else:
        print("No ground truth trajectories found.")
else:
    print("Future prediction not found. Run 04_future_prediction.py first.")

print("Trajectory difference evaluation completed.")