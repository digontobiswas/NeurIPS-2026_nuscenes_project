import os
import torch
import numpy as np

print("CausalCoop-WM Reward Function")
print("==================================================")

world_model_dir = "outputs/world_model_data"
v2x_dir = "outputs/v2x_cooperative"
output_dir = "outputs/evaluation"
os.makedirs(output_dir, exist_ok=True)

future_path = os.path.join(world_model_dir, "future_prediction.pt")
fused_path = os.path.join(v2x_dir, "fused_belief.pt")

if os.path.exists(future_path):
    predicted_positions = torch.load(future_path)
    
    # Simple safety + efficiency reward
    collision_penalty = 0.0
    progress_reward = 0.0
    
    # Check for potential collisions in future trajectory
    for i in range(1, len(predicted_positions)):
        dist_moved = np.linalg.norm(np.array(predicted_positions[i]) - np.array(predicted_positions[i-1]))
        progress_reward += dist_moved * 0.1
        
        if dist_moved < 0.5:  # sudden stop or collision risk
            collision_penalty -= 5.0
    
    total_reward = progress_reward + collision_penalty
    
    results = {
        "total_reward": float(total_reward),
        "progress_reward": float(progress_reward),
        "collision_penalty": float(collision_penalty),
        "num_future_frames": len(predicted_positions)
    }
    
    torch.save(results, os.path.join(output_dir, "reward_results.pt"))
    print("Total reward: " + str(round(total_reward, 4)))
    print("Progress reward: " + str(round(progress_reward, 4)))
    print("Collision penalty: " + str(round(collision_penalty, 4)))
    print("Reward function results saved to outputs/evaluation/reward_results.pt")
else:
    print("Future prediction not found. Run world model scripts first.")

print("Reward function evaluation completed.")