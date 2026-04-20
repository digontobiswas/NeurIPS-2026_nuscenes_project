import os
from nuscenes.nuscenes import NuScenes

print("CausalCoop-WM Sanity Check")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"

if not os.path.exists(data_root):
    print("Data root not found. Please download nuScenes mini and place in data_nuscenes")
else:
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    print("NuScenes loaded successfully")
    print("Number of scenes: " + str(len(nusc.scene)))
    print("Number of samples: " + str(len(nusc.sample)))
    print("Number of annotations: " + str(len(nusc.sample_annotation)))
    print("Dataset version: " + version)
    print("Sanity check completed. Dataset is ready.")