import os
from nuscenes.nuscenes import NuScenes

print("CausalCoop-WM Agent Exploration")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

scene_index = 0
scene = nusc.scene[scene_index]
print("Exploring scene: " + scene["name"])

sample_token = scene["first_sample_token"]
sample = nusc.get("sample", sample_token)

print("Number of agents in this sample: " + str(len(sample["anns"])))

for i, ann_token in enumerate(sample["anns"][:10]):
    ann = nusc.get("sample_annotation", ann_token)
    print("Agent " + str(i) + ":")
    print("   Category: " + ann["category_name"])
    print("   Translation: " + str(ann["translation"]))
    print("   Size: " + str(ann["size"]))
    print("   Velocity: " + str(ann.get("velocity", "N/A")))

print("Agent exploration completed for first 10 annotations.")