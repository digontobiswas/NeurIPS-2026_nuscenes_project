import os
import matplotlib.pyplot as plt

print("CausalCoop-WM Generate All Figures")
print("==================================================")

figure_dir = "outputs/figures"
os.makedirs(figure_dir, exist_ok=True)

print("Generating summary figure layout...")

# Create a combined summary figure
fig = plt.figure(figsize=(20, 12))

# Placeholder for multiple subplots (will be populated as more outputs become available)
plt.subplot(2, 3, 1)
plt.text(0.5, 0.5, "Trajectories Plot\n(see 01_plot_trajectories.py)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.subplot(2, 3, 2)
plt.text(0.5, 0.5, "Causal Graph\n(see 02_plot_causal_graph.py)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.subplot(2, 3, 3)
plt.text(0.5, 0.5, "Attention Map\n(see 03_plot_attention_maps.py)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.subplot(2, 3, 4)
plt.text(0.5, 0.5, "LiDAR Visualization\n(from exploration)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.subplot(2, 3, 5)
plt.text(0.5, 0.5, "Camera Views\n(from exploration)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.subplot(2, 3, 6)
plt.text(0.5, 0.5, "Evaluation Metrics\n(FID/FVD, Reward, etc.)", ha="center", va="center", fontsize=14)
plt.axis("off")

plt.suptitle("CausalCoop-WM Project - All Generated Figures Overview", fontsize=16)
plt.tight_layout()

out_path = os.path.join(figure_dir, "summary_all_figures.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Summary figure saved to " + out_path)
print("All figures generation completed.")