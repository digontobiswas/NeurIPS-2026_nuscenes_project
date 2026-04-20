import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATAROOT   = 'D:/nuscenes_project/data/nuscenes'
OUTPUT_DIR = 'outputs/world_model_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)


class NuScenesSequenceDataset(Dataset):
    """
    Simple dataset that loads trajectory sequences
    from saved pkl files for world model training.
    """

    def __init__(self, trajectory_dir, seq_len=10):
        self.seq_len = seq_len
        self.samples = []

        traj_files = [
            f for f in os.listdir(trajectory_dir)
            if f.endswith('.pkl')
        ]

        for fname in traj_files:
            path = os.path.join(trajectory_dir, fname)
            with open(path, 'rb') as f:
                trajectories = pickle.load(f)

            for inst_token, traj in trajectories.items():
                if len(traj) >= seq_len:
                    for start in range(len(traj) - seq_len + 1):
                        window = traj[start:start + seq_len]
                        self.samples.append({
                            'instance': inst_token,
                            'sequence': window,
                            'category': traj[0]['category']
                        })

        print(f"Dataset loaded: {len(self.samples)} sequences")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        seq  = item['sequence']

        positions = torch.tensor(
            [[p['x'], p['y'], p['z']] for p in seq],
            dtype=torch.float32
        )

        if len(positions) > 1:
            velocities = torch.diff(positions, dim=0)
            velocities = torch.cat(
                [velocities[:1], velocities], dim=0
            )
        else:
            velocities = torch.zeros_like(positions)

        return {
            'positions' : positions,
            'velocities': velocities,
            'category'  : item['category'],
            'instance'  : item['instance']
        }


if __name__ == '__main__':
    trajectory_dir = 'outputs/trajectories'

    if not os.path.exists(trajectory_dir) or \
       len(os.listdir(trajectory_dir)) == 0:
        print("No trajectories found.")
        print("Run 01_exploration/03_extract_trajectories.py first.")
        exit()

    dataset    = NuScenesSequenceDataset(trajectory_dir, seq_len=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print()
    print('DATALOADER TEST:')
    print('-' * 40)
    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        print(f"  Positions shape  : {batch['positions'].shape}")
        print(f"  Velocities shape : {batch['velocities'].shape}")
        print(f"  Categories       : {batch['category']}")
        if batch_idx >= 2:
            break

    out_path = os.path.join(OUTPUT_DIR, 'dataset_info.pkl')
    info = {
        'total_sequences': len(dataset),
        'seq_len'        : 5
    }
    with open(out_path, 'wb') as f:
        pickle.dump(info, f)
    print(f"\nDataset info saved: {out_path}")