import os
import torch
import torch.nn as nn

INPUT_DIR  = 'outputs/v2x_cooperative'
OUTPUT_DIR = 'outputs/v2x_cooperative'

os.makedirs(OUTPUT_DIR, exist_ok=True)


class BeliefFusionModule(nn.Module):
    """
    Fuses compressed belief vectors from multiple agents
    using attention-based aggregation.
    This is the core PS-2 module.
    """
    def __init__(self, belief_dim=16, n_heads=4,
                 output_dim=128):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=belief_dim,
            num_heads=n_heads,
            batch_first=True
        )
        self.norm  = nn.LayerNorm(belief_dim)
        self.proj  = nn.Linear(belief_dim, output_dim)
        self.relu  = nn.ReLU()

    def forward(self, beliefs):
        """
        beliefs: (N, belief_dim) — one belief per agent
        Returns fused belief: (output_dim,)
        """
        x = beliefs.unsqueeze(0)

        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(attn_out + x)

        pooled = x.mean(dim=1)
        fused  = self.relu(self.proj(pooled.squeeze(0)))

        return fused, attn_weights


beliefs_path = os.path.join(INPUT_DIR, 'compressed_beliefs.pt')

if not os.path.exists(beliefs_path):
    print("Compressed beliefs not found.")
    print("Run 02_latent_belief_compress.py first.")
    exit()

beliefs = torch.load(beliefs_path, weights_only=False)

print(f"Loaded beliefs shape: {beliefs.shape}")
print(f"  Agents    : {beliefs.shape[0]}")
print(f"  Belief dim: {beliefs.shape[1]}")
print()

fusion_module = BeliefFusionModule(
    belief_dim=beliefs.shape[1],
    n_heads=2,
    output_dim=128
)
fusion_module.eval()

with torch.no_grad():
    fused_belief, attn_weights = fusion_module(beliefs)

print(f"Fused belief shape   : {fused_belief.shape}")
print(f"Attention weights    : {attn_weights.shape}")
print()
print(f"Fused belief stats:")
print(f"  Mean : {fused_belief.mean().item():.4f}")
print(f"  Std  : {fused_belief.std().item():.4f}")
print(f"  Min  : {fused_belief.min().item():.4f}")
print(f"  Max  : {fused_belief.max().item():.4f}")

out_path = os.path.join(OUTPUT_DIR, 'fused_belief.pt')
torch.save(fused_belief, out_path)
print(f"\nFused belief saved: {out_path}")

attn_path = os.path.join(OUTPUT_DIR, 'attention_weights.pt')
torch.save(attn_weights, attn_path)
print(f"Attention weights saved: {attn_path}")