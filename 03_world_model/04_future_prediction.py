import os
import torch
import numpy as np

print("CausalCoop-WM Future Prediction")
print("==================================================")

data_dir = "outputs/world_model_data"

# Load sequence and modified latents
sequence_path = os.path.join(data_dir, "sample_sequence_0.pt")
latents_path = os.path.join(data_dir, "modified_latents.pt")

if os.path.exists(sequence_path) and os.path.exists(latents_path):
    frame_data = torch.load(sequence_path)
    latents = torch.load(latents_path)
    
    print("Loaded sequence with " + str(len(frame_data)) + " frames")
    print("Loaded latents shape: " + str(latents.shape))
    
    # Simulate future prediction (placeholder for Vista decoder)
    # In full implementation this will call Vista diffusion decoder
    future_frames = 8
    predicted_positions = []
    
    # Simple linear extrapolation using ego pose for demo
    ego_start = np.array(frame_data[-1]["ego_pose"])
    for i in range(future_frames):
        future_pos = ego_start + np.array([i * 2.0, 0.0, 0.0])  # move forward
        predicted_positions.append(future_pos.tolist())
    
    torch.save(predicted_positions, os.path.join(data_dir, "future_prediction.pt"))
    print("Future prediction generated for " + str(future_frames) + " frames")
    print("Future prediction saved to outputs/world_model_data/future_prediction.pt")
else:
    print("Required files not found. Run previous scripts in this folder first.")

print("Future prediction completed.")