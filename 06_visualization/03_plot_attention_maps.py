import os
import torch
import numpy as np
import matplotlib.pyplot as plt

ATTN_PATH  = 'outputs/v2x_cooperative/attention_weights.pt'
OUTPUT_DIR = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(ATTN_PATH):
    print("Attention weights not found.")
    print("Run 04_v2x_cooperative/03_belief_fusion.py first.")
    exit()

attn = torch.load(ATTN_PATH, weights_only=False)
print(f"Attention weights shape: {attn.shape}")

if attn.dim() == 3:
    attn_np = attn[0].detach().numpy()
elif attn.dim() == 2:
    attn_np = attn.detach().numpy()
else:
    attn_np = attn.squeeze().detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('V2X Attention Weights — Agent Belief Fusion', fontsize=13)

im = axes[0].imshow(attn_np, cmap='Blues', aspect='auto',
                    vmin=0, vmax=attn_np.max())
plt.colorbar(im, ax=axes[0])
axes[0].set_title('Attention Heatmap\n(rows=query, cols=key)')
axes[0].set_xlabel('Key agent')
axes[0].set_ylabel('Query agent')
n = attn_np.shape[0]
axes[0].set_xticks(range(n))
axes[0].set_yticks(range(n))
axes[0].set_xticklabels([f'A{i}' for i in range(n)])
axes[0].set_yticklabels([f'A{i}' for i in range(n)])

row_sums = attn_np.sum(axis=1)
axes[1].bar(range(len(row_sums)), row_sums,
            color='steelblue', alpha=0.8)
axes[1].set_title('Attention Sum per Query Agent')
axes[1].set_xlabel('Query agent')
axes[1].set_ylabel('Sum of attention weights')
axes[1].set_xticks(range(len(row_sums)))
axes[1].set_xticklabels([f'A{i}' for i in range(len(row_sums))])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'attention_weights.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
plt.close()