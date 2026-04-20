import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from PIL import Image

print("CausalCoop-WM Camera Visualization")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

scene_index = 0
sample_index = 0
scene = nusc.scene[scene_index]
sample_token = scene["first_sample_token"]
for _ in range(sample_index):
    sample = nusc.get("sample", sample_token)
    sample_token = sample["next"]
sample = nusc.get("sample", sample_token)

camera_tokens = [sample["data"][cam] for cam in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, token in enumerate(camera_tokens):
    cam_data = nusc.get("sample_data", token)
    im = Image.open(cam_data["filename"])
    axes[i].imshow(im)
    axes[i].set_title(cam_data["channel"])
    axes[i].axis("off")

plt.tight_layout()
output_dir = "outputs/figures"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "camera_views_scene" + scene["token"][:8] + ".png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()

print("Camera visualization saved to " + out_path)
print("Camera visualization completed.")