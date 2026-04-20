import os
import torch

print("CausalCoop-WM Latent Belief Compression")
print("==================================================")

world_model_dir = "outputs/world_model_data"
v2x_dir = "outputs/v2x_cooperative"
os.makedirs(v2x_dir, exist_ok=True)

latent_path = os.path.join(world_model_dir, "modified_latents.pt")
if os.path.exists(latent_path):
    latents = torch.load(latent_path)
    print("Loaded latents with shape " + str(latents.shape))
    
    # Compression for bandwidth constraint
    compression_ratio = 0.15
    original_dim = latents.shape[-1]
    compressed_dim = int(original_dim * compression_ratio)
    
    # Simple linear compression (in full model this will use learned encoder)
    compressed_beliefs = []
    for i in range(latents.shape[0]):
        agent_latent = latents[i]
        compressed = torch.nn.functional.adaptive_avg_pool1d(agent_latent.unsqueeze(0), compressed_dim).squeeze(0)
        compressed_beliefs.append(compressed)
    
    compressed_beliefs = torch.stack(compressed_beliefs)
    
    torch.save(compressed_beliefs, os.path.join(v2x_dir, "compressed_beliefs.pt"))
    print("Latent beliefs compressed from dim " + str(original_dim) + " to " + str(compressed_dim))
    print("Compressed beliefs saved to outputs/v2x_cooperative/compressed_beliefs.pt")
else:
    print("Latents not found. Run world model scripts first.")

print("Latent belief compression completed.")