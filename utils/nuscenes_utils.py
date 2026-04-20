import os
from nuscenes.nuscenes import NuScenes

def load_nuscenes_scene(data_root, version="v1.0-mini", scene_index=0):
    print("Loading nuScenes scene")
    nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
    scene = nusc.scene[scene_index]
    return nusc, scene

def get_multi_agent_data(nusc, sample_token, num_agents=4):
    print("Getting multi-agent data")
    sample = nusc.get("sample", sample_token)
    multi_agent_data = []
    
    for agent_id in range(num_agents):
        agent_data = {
            "agent_id": agent_id,
            "sample_token": sample_token,
            "anns": sample["anns"][:10] if len(sample["anns"]) > 0 else [],
            "ego_pose": nusc.get("ego_pose", sample["data"]["CAM_FRONT"]["ego_pose_token"])["translation"]
        }
        multi_agent_data.append(agent_data)
    return multi_agent_data

def extract_sequence(nusc, scene, sequence_length=16):
    print("Extracting sequence data")
    sample_token = scene["first_sample_token"]
    frame_data = []
    
    for i in range(sequence_length):
        if sample_token == "":
            break
        sample = nusc.get("sample", sample_token)
        
        cam_token = sample["data"]["CAM_FRONT"]
        cam_data = nusc.get("sample_data", cam_token)
        
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = nusc.get("sample_data", lidar_token)
        
        frame_data.append({
            "frame_idx": i,
            "sample_token": sample_token,
            "ego_pose": nusc.get("ego_pose", cam_data["ego_pose_token"])["translation"]
        })
        
        sample_token = sample["next"]
    
    return frame_data