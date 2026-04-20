import os
import torch

print("CausalCoop-WM Belief Fusion")
print("==================================================")

v2x_dir = "outputs/v2x_cooperative"
os.makedirs(v2x_dir, exist_ok=True)

compressed_path = os.path.join(v2x_dir, "compressed_beliefs.pt")
setup_path = os.path.join(v2x_dir, "multi_agent_setup.pt")

if os.path.exists(compressed_path) and os.path.exists(setup_path):
    compressed_beliefs = torch.load(compressed_path)
    multi_agent_data = torch.load(setup_path)
    
    print("Loaded compressed beliefs for " + str(len(compressed_beliefs)) + " agents")
    
    # Belief fusion: simple average fusion across agents
    fused_belief = torch.mean(compressed_beliefs, dim=0)
    
    # Save fused belief
    torch.save(fused_belief, os.path.join(v2x_dir, "fused_belief.pt"))
    print("Beliefs fused across agents")
    print("Fused belief shape: " + str(fused_belief.shape))
    print("Fused belief saved to outputs/v2x_cooperative/fused_belief.pt")
else:
    print("Required files not found. Run previous V2X scripts first.")

print("Belief fusion completed.")