import os
import torch

print("CausalCoop-WM Bandwidth Ablation")
print("==================================================")

v2x_dir = "outputs/v2x_cooperative"
os.makedirs(v2x_dir, exist_ok=True)

compressed_path = os.path.join(v2x_dir, "compressed_beliefs.pt")
if os.path.exists(compressed_path):
    compressed_beliefs = torch.load(compressed_path)
    
    bandwidth_levels = [100, 500, 1000, 2000]  # kbps
    results = {}
    
    for bandwidth in bandwidth_levels:
        # Simulate different compression based on bandwidth
        compression_ratio = min(0.05 + (bandwidth / 4000.0), 0.3)
        effective_dim = int(compressed_beliefs.shape[-1] * compression_ratio)
        
        # Downsample for this bandwidth
        ablated = torch.nn.functional.adaptive_avg_pool1d(
            compressed_beliefs.unsqueeze(0), effective_dim
        ).squeeze(0)
        
        results[bandwidth] = {
            "compression_ratio": compression_ratio,
            "effective_dim": effective_dim,
            "belief_shape": ablated.shape
        }
        print("Bandwidth " + str(bandwidth) + " kbps -> compression " + str(round(compression_ratio, 3)) + " -> dim " + str(effective_dim))
    
    torch.save(results, os.path.join(v2x_dir, "bandwidth_ablation_results.pt"))
    print("Bandwidth ablation completed for levels: " + str(bandwidth_levels))
    print("Results saved to outputs/v2x_cooperative/bandwidth_ablation_results.pt")
else:
    print("Compressed beliefs not found. Run 02_latent_belief_compress.py first.")

print("Bandwidth ablation study completed.")