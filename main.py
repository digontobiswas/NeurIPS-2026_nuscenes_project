import os
import yaml
import argparse
import torch
import pickle
import numpy as np
from nuscenes.nuscenes import NuScenes

from utils import (
    load_nuscenes_scene,
    get_multi_agent_data,
    extract_sequence,
    build_causal_graph,
    infer_agent_intent,
    run_counterfactual_query,
    compute_fid_fvd,
    compute_trajectory_difference,
    compute_reward,
    plot_trajectories,
    plot_causal_graph,
    plot_attention_map
)

print("CausalCoop-WM Project Main Entry")
print("=" * 50)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def ensure_directories():
    dirs = [
        "outputs",
        "outputs/trajectories",
        "outputs/causal_graphs",
        "outputs/world_model_data",
        "outputs/v2x_cooperative",
        "outputs/evaluation",
        "outputs/figures",
        "outputs/logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("All output directories ready.")


def run_exploration(nusc, scene, output_dir):
    """Extract and save trajectories from nuScenes."""
    from collections import defaultdict

    trajectories = defaultdict(list)
    sample_token = scene['first_sample_token']

    while sample_token:
        sample = nusc.get('sample', sample_token)
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            x, y, z = ann['translation']
            trajectories[ann['instance_token']].append({
                'x'        : x,
                'y'        : y,
                'z'        : z,
                'timestamp': sample['timestamp'],
                'category' : ann['category_name'],
                'num_lidar': ann['num_lidar_pts']
            })
        sample_token = sample['next']

    out_path = os.path.join(output_dir, 'trajectories_scene0.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dict(trajectories), f)

    print(f"Trajectories saved: {out_path}")
    print(f"Total agents: {len(trajectories)}")
    return dict(trajectories)


def main():
    parser = argparse.ArgumentParser(
        description='CausalCoop-WM Pipeline'
    )
    parser.add_argument(
        '--stage', type=str, default='all',
        choices=[
            'all', 'exploration', 'causal',
            'world_model', 'v2x', 'evaluation', 'visualization'
        ]
    )
    args = parser.parse_args()

    # Load configs
    base_config   = load_config('configs/base_config.yaml')
    causal_config = load_config('configs/causal_config.yaml')
    v2x_config    = load_config('configs/v2x_config.yaml')

    data_root = base_config['data_root']
    version   = base_config.get('version', 'v1.0-mini')

    print(f"Data root : {data_root}")
    print(f"Version   : {version}")
    print(f"Stage     : {args.stage}")
    print()

    ensure_directories()

    trajectory_dir = 'outputs/trajectories'
    graph_path     = causal_config['causal']['output_graph']
    intent_path    = causal_config['causal']['output_intents']
    cf_path        = causal_config['causal']['output_counterfactual']

    # ── 01 Exploration ───────────────────────────────────────
    if args.stage in ['all', 'exploration']:
        print('\n=== Phase 1: Exploration ===')
        nusc, scene = load_nuscenes_scene(data_root, version)
        run_exploration(nusc, scene, trajectory_dir)
        sequence = extract_sequence(nusc, scene)
        print('Exploration phase complete.\n')

    # ── 02 Causal Model ──────────────────────────────────────
    if args.stage in ['all', 'causal']:
        print('\n=== Phase 2: Causal Model ===')
        G           = build_causal_graph(trajectory_dir, graph_path)
        intent_data = infer_agent_intent(trajectory_dir, intent_path)
        cf_result   = run_counterfactual_query(graph_path, intent_path, cf_path)
        print('Causal model phase complete.\n')

    # ── 03 World Model ───────────────────────────────────────
    if args.stage in ['all', 'world_model']:
        print('\n=== Phase 3: World Model ===')
        wm_dir = 'outputs/world_model_data'
        os.makedirs(wm_dir, exist_ok=True)

        # Create dummy tensors for now (replace with real model output)
        real_seq = torch.randn(25, 3, 64, 64)
        pred_seq = torch.randn(25, 3, 64, 64)

        torch.save(real_seq, os.path.join(wm_dir, 'sample_sequence_0.pt'))
        torch.save(pred_seq, os.path.join(wm_dir, 'future_prediction.pt'))
        print(f"World model tensors saved to {wm_dir}")
        print('World model phase complete.\n')

    # ── 04 V2X Cooperative ───────────────────────────────────
    if args.stage in ['all', 'v2x']:
        print('\n=== Phase 4: V2X Cooperative ===')
        v2x_dir = v2x_config['v2x']['output_dir']
        os.makedirs(v2x_dir, exist_ok=True)

        latent_dim  = v2x_config['v2x']['latent_dim']
        num_agents  = v2x_config['v2x']['num_agents']
        fused_belief = torch.randn(num_agents, latent_dim)

        fused_path = v2x_config['v2x']['fused_belief']
        torch.save(fused_belief, fused_path)
        print(f"Fused belief saved: {fused_path}")
        print('V2X phase complete.\n')

    # ── 05 Evaluation ────────────────────────────────────────
    if args.stage in ['all', 'evaluation']:
        print('\n=== Phase 5: Evaluation ===')
        seq_path    = 'outputs/world_model_data/sample_sequence_0.pt'
        future_path = 'outputs/world_model_data/future_prediction.pt'

        if os.path.exists(seq_path) and os.path.exists(future_path):
            real_seq = torch.load(seq_path,    weights_only=False)
            pred_seq = torch.load(future_path, weights_only=False)

            scores = compute_fid_fvd(real_seq, pred_seq)
            print(f"FID  : {scores['fid']:.4f}")
            print(f"FVD  : {scores['fvd']:.4f}")

            gt_pos   = [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]]
            pred_pos = [[0, 0], [2.1, 0.1], [4.2, 0.1], [6.1, 0.2], [8.3, 0.1]]
            traj     = compute_trajectory_difference(gt_pos, pred_pos)
            print(f"RMSE : {traj['rmse']:.4f}")
            print(f"MAE  : {traj['mae']:.4f}")

            reward = compute_reward(pred_pos)
            print(f"Reward: {reward['total_reward']:.4f}")
        else:
            print("Run world_model stage first.")
        print('Evaluation phase complete.\n')

    # ── 06 Visualization ─────────────────────────────────────
    if args.stage in ['all', 'visualization']:
        print('\n=== Phase 6: Visualization ===')
        plot_trajectories(trajectory_dir)

        if os.path.exists(graph_path):
            plot_causal_graph(graph_path)
        else:
            print("Causal graph not found — run causal stage first.")

        fused_path = v2x_config['v2x']['fused_belief']
        if os.path.exists(fused_path):
            plot_attention_map(fused_path)
        else:
            print("Fused belief not found — run v2x stage first.")
        print('Visualization phase complete.\n')

    print('=' * 50)
    print('Pipeline finished. Outputs saved in outputs/')


if __name__ == '__main__':
    main()