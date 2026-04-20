import os
import torch
from nuscenes.nuscenes import NuScenes

print("CausalCoop-WM Multi Agent Setup")
print("==================================================")

data_root = r"D:\nuscenes_project\data\nuscenes"
version = "v1.0-mini"
nusc = NuScenes(version=version, dataroot=data_root, verbose=True)

output_dir = "outputs/v2x_cooperative"
os.makedirs(output_dir, exist_ok=True)

num_agents = 4
scene_index = 0
scene = nusc.scene[scene_index]
sample_token = scene["first_sample_token"]

multi_agent_data = []

for agent_id in range(num_agents):
    sample = nusc.get("sample", sample_token)
    agent_data = {
        "agent_id": agent_id,
        "sample_token": sample_token,
        "anns": sample["anns"][:10] if len(sample["anns"]) > 0 else [],
        "ego_pose": nusc.get("ego_pose", sample["data"]["CAM_FRONT"]["ego_pose_token"])["translation"]
    }
    multi_agent_data.append(agent_data)
    print("Agent " + str(agent_id) + " setup complete with " + str(len(agent_data["anns"])) + " annotations")

torch.save(multi_agent_data, os.path.join(output_dir, "multi_agent_setup.pt"))
print("Multi-agent setup completed for " + str(num_agents) + " agents")
print("Data saved to outputs/v2x_cooperative/multi_agent_setup.pt")