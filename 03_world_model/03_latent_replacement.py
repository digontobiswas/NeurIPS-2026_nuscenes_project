import os
import torch

print("CausalCoop-WM Latent Replacement")
print("==================================================")

data_dir = "outputs/world_model_data"
output_dir = "outputs/world_model_data"
os.makedirs(output_dir, exist_ok=True)

# Load sample sequence
sequence_path = os.path.join(data_dir, "sample_sequence_0.pt")
if os.path.exists(sequence_path):
    frame_data = torch.load(sequence_path)
    print("Loaded sequence with " + str(len(frame_data)) + " frames")
    
    # Simulate latent extraction (placeholder for Vista latents)
    # In full Vista integration this will use the actual encoder
    batch_size = 1
    latent_dim = 64
    num_frames = len(frame_data)
    
    # Dummy latents for now (will be replaced with Vista encoder later)
    latents = torch.randn(batch_size, num_frames, latent_dim)
    
    # Latent replacement for causal intervention
    # Example: replace latent of agent at frame 8
    intervention_frame = 8
    if intervention_frame < num_frames:
        latents[0, intervention_frame] = torch.zeros(latent_dim)  # zero out for intervention
    
    torch.save(latents, os.path.join(output_dir, "modified_latents.pt"))
    print("Latent replacement performed at frame " + str(intervention_frame))
    print("Modified latents saved to outputs/world_model_data/modified_latents.pt")
else:
    print("Sequence not found. Run 01_data_loader.py first.")

print("Latent replacement completed.")