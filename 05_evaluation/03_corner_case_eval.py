import os
import torch
import numpy as np
from nuscenes.nuscenes import NuScenes

print("CausalCoop-WM Corner Case Evaluation")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

output_dir = "outputs/evaluation"
os.makedirs(output_dir, exist_ok=True)

scene_index = 0
scene = nusc.scene[scene_index]
sample_token = scene["first_sample_token"]
sample = nusc.get("sample", sample_token)

# Detect corner cases (e.g. close proximity agents)
corner_cases = 0
for ann_token in sample["anns"][:20]:
    ann = nusc.get("sample_annotation", ann_token)
    dist_to_ego = np.linalg.norm(np.array(ann["translation"]) - np.array([0, 0, 0]))
    if dist_to_ego < 8.0 and ann["category_name"].startswith("vehicle"):
        corner_cases += 1
        print("Corner case detected: " + ann["category_name"] + " at " + str(round(dist_to_ego, 2)) + "m")

results = {
    "corner_cases_detected": corner_cases,
    "total_agents_checked": min(20, len(sample["anns"])),
    "corner_case_rate": corner_cases / max(1, min(20, len(sample["anns"])))
}

torch.save(results, os.path.join(output_dir, "corner_case_results.pt"))
print("Corner cases detected: " + str(corner_cases))
print("Corner case rate: " + str(round(results["corner_case_rate"], 4)))
print("Corner case evaluation saved to outputs/evaluation/corner_case_results.pt")
print("Corner case evaluation completed.")