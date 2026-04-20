import os
import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image

print("CausalCoop-WM Data Loader")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

output_dir = "outputs/world_model_data"
os.makedirs(output_dir, exist_ok=True)

scene_index = 0
scene = nusc.scene[scene_index]
sample_token = scene["first_sample_token"]

sequence_length = 16
frame_data = []

for i in range(sequence_length):
    if sample_token == "":
        break
    sample = nusc.get("sample", sample_token)
    
    # Camera front image
    cam_token = sample["data"]["CAM_FRONT"]
    cam_data = nusc.get("sample_data", cam_token)
    img = Image.open(cam_data["filename"]).resize((512, 256))
    img_array = np.array(img) / 255.0
    
    # LiDAR point cloud
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    points = np.fromfile(lidar_data["filename"], dtype=np.float32).reshape(-1, 5)[:, :3]
    
    # Ego pose
    ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
    
    frame_data.append({
        "frame_idx": i,
        "image": img_array,
        "lidar_points": points,
        "ego_pose": ego_pose["translation"],
        "sample_token": sample_token
    })
    
    sample_token = sample["next"]

# Save sequence
torch.save(frame_data, os.path.join(output_dir, "sample_sequence_0.pt"))
print("Data sequence loaded with " + str(len(frame_data)) + " frames")
print("Sequence saved to outputs/world_model_data/sample_sequence_0.pt")
print("Data loader completed.")