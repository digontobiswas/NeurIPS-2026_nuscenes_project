import os
import torch
import matplotlib.pyplot as plt

print("CausalCoop-WM Plot Attention Maps")
print("==================================================")

v2x_dir = "outputs/v2x_cooperative"
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)

fused_path = os.path.join(v2x_dir, "fused_belief.pt")

if os.path.exists(fused_path):
    fused_belief = torch.load(fused_path)
    
    # Simulate attention map (placeholder for future transformer attention)
    attention_map = torch.softmax(fused_belief[:64].unsqueeze(0), dim=1).squeeze(0).reshape(8, 8).cpu().numpy()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map, cmap="viridis")
    plt.colorbar(label="Attention Weight")
    plt.title("Simulated Attention Map (Belief Fusion)")
    plt.xlabel("Latent Dimension X")
    plt.ylabel("Latent Dimension Y")
    
    out_path = os.path.join(output_dir, "attention_map_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    
    print("Attention map plot saved to " + out_path)
else:
    print("Fused belief not found. Run V2X scripts first.")

print("Attention map plotting completed.")