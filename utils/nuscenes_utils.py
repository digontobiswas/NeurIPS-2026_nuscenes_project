import os
import pickle
import numpy as np
from collections import defaultdict
from nuscenes.nuscenes import NuScenes


def load_nuscenes_scene(data_root, version='v1.0-mini', scene_index=0):
    """
    Load nuScenes dataset and return the dataset object
    and a specific scene.
    """
    nusc = NuScenes(
        version=version,
        dataroot=data_root,
        verbose=False
    )
    scene = nusc.scene[scene_index]
    print(f"Loaded scene: {scene['name']}")
    print(f"Description : {scene['description']}")
    print(f"Frames      : {scene['nbr_samples']}")
    return nusc, scene


def get_multi_agent_data(nusc, scene):
    """
    Extract all agent data from a scene.
    Returns a dict of instance_token -> list of frame data.
    """
    agents = defaultdict(list)
    sample_token = scene['first_sample_token']

    while sample_token:
        sample = nusc.get('sample', sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            x, y, z = ann['translation']
            agents[ann['instance_token']].append({
                'x'          : x,
                'y'          : y,
                'z'          : z,
                'timestamp'  : sample['timestamp'],
                'category'   : ann['category_name'],
                'num_lidar'  : ann['num_lidar_pts'],
                'ann_token'  : ann_token,
                'sample_token': sample_token
            })
        sample_token = sample['next']

    return dict(agents)


def extract_sequence(nusc, scene, output_dir='outputs/world_model_data'):
    """
    Extract camera image paths and LiDAR paths for
    every keyframe in the scene as a sequence.
    Returns a list of dicts, one per keyframe.
    """
    os.makedirs(output_dir, exist_ok=True)
    sequence = []
    sample_token = scene['first_sample_token']
    frame_idx = 0

    cameras = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    while sample_token:
        sample = nusc.get('sample', sample_token)
        frame_data = {
            'frame_idx'  : frame_idx,
            'timestamp'  : sample['timestamp'],
            'sample_token': sample_token,
            'cameras'    : {},
            'lidar_path' : None
        }

        # Camera paths
        for cam in cameras:
            if cam in sample['data']:
                cam_data = nusc.get('sample_data', sample['data'][cam])
                frame_data['cameras'][cam] = cam_data['filename']

        # LiDAR path
        if 'LIDAR_TOP' in sample['data']:
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            frame_data['lidar_path'] = lidar_data['filename']

        sequence.append(frame_data)
        frame_idx   += 1
        sample_token = sample['next']

    # Save sequence
    out_path = os.path.join(output_dir, 'scene_sequence.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(sequence, f)
    print(f"Sequence saved: {out_path} ({len(sequence)} frames)")

    return sequence