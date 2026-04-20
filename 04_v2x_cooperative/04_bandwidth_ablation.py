import os
import torch
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR  = 'outputs/v2x_cooperative'
OUTPUT_DIR = 'outputs/figures'

os.makedirs(OUTPUT_DIR, exist_ok=True)

beliefs_path = os.path.join(INPUT_DIR, 'compressed_beliefs.pt')

if not os.path.exists(beliefs_path):
    print("Compressed beliefs not found.")
    print("Run 02_latent_belief_compress.py first.")
    exit()

beliefs = torch.load(beliefs_path, weights_only=False)

print(f"Running bandwidth ablation on {beliefs.shape[0]} agents...")
print(f"Full belief dim: {beliefs.shape[1]}")
print()


def apply_bandwidth_limit(beliefs, keep_ratio):
    """Simulate bandwidth limit by keeping top-k dimensions."""
    n_keep  = max(1, int(beliefs.shape[1] * keep_ratio))
    limited = beliefs.clone()
    limited[:, n_keep:] = 0.0
    return limited


def compute_reconstruction_error(original, limited):
    """MSE between original and bandwidth-limited beliefs."""
    return float(torch.mean((original - limited) ** 2).item())


bandwidth_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
errors           = []
dims_used        = []

print(f"{'Bandwidth':<15} {'Dims used':<12} {'MSE Error':<15} {'Quality'}")
print('-' * 55)

for ratio in bandwidth_ratios:
    limited = apply_bandwidth_limit(beliefs, ratio)
    error   = compute_reconstruction_error(beliefs, limited)
    n_dims  = max(1, int(beliefs.shape[1] * ratio))
    quality = 'Excellent' if error < 0.01 else \
              'Good'      if error < 0.1  else \
              'Fair'      if error < 0.5  else 'Poor'

    errors.append(error)
    dims_used.append(n_dims)

    print(
        f"  {ratio*100:.0f}%{'':<10} "
        f"{n_dims:<12} "
        f"{error:<15.6f} "
        f"{quality}"
    )

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('V2X Bandwidth Ablation Study', fontsize=13)

axes[0].plot(
    [r * 100 for r in bandwidth_ratios],
    errors, 'b-o', linewidth=2, markersize=6
)
axes[0].set_title('Reconstruction Error vs Bandwidth')
axes[0].set_xlabel('Bandwidth used (%)')
axes[0].set_ylabel('MSE Reconstruction Error')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=0.1, color='r', linestyle='--',
                label='Acceptable threshold')
axes[0].legend()

axes[1].bar(
    [r * 100 for r in bandwidth_ratios],
    dims_used,
    color='steelblue', alpha=0.8, width=7
)
axes[1].set_title('Belief Dimensions Transmitted')
axes[1].set_xlabel('Bandwidth used (%)')
axes[1].set_ylabel('Dimensions transmitted')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'bandwidth_ablation.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved: {out_path}")
plt.show()
plt.close()

results = {
    'bandwidth_ratios': bandwidth_ratios,
    'errors'          : errors,
    'dims_used'       : dims_used
}
import pickle
with open(os.path.join(INPUT_DIR, 'bandwidth_ablation.pkl'), 'wb') as f:
    pickle.dump(results, f)
print("Ablation results saved.")