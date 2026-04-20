import os
import yaml
import argparse
import torch
from nuscenes.nuscenes import NuScenes

# Import custom utils
from utils import load_nuscenes_scene, get_multi_agent_data, extract_sequence
from utils import build_causal_graph, infer_agent_intent, run_counterfactual_query
from utils import compute_fid_fvd, compute_trajectory_difference, compute_reward
from utils import plot_trajectories, plot_causal_graph, plot_attention_map

print("CausalCoop-WM Project Main Entry")
print("==================================================")

def load_config(config_path):
    with open(config_path, "r") as f:
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
    print("All output directories created.")

def run_exploration_if_needed():
    trajectory_dir = "outputs/trajectories"
    if not os.path.exists(trajectory_dir) or len([f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]) == 0:
        print("\n=== Running Trajectory Extraction (first time) ===")
        exec(open("01_exploration/03_extract_trajectories.py").read())
        print("Trajectory extraction completed.")
    else:
        print("Trajectories already exist - skipping extraction.")

def main():
    parser = argparse.ArgumentParser(description="CausalCoop-WM Full Pipeline Runner")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "exploration", "causal", "world_model", "v2x", "evaluation", "visualization"])
    args = parser.parse_args()

    base_config = load_config("configs/base_config.yaml")
    causal_config = load_config("configs/causal_config.yaml")
    v2x_config = load_config("configs/v2x_config.yaml")

    data_root = base_config["data_root"]
    print("Data root: " + data_root)
    print("Running stage: " + args.stage)

    ensure_directories()

    # 01 Exploration Phase
    if args.stage in ["all", "exploration"]:
        print("\n=== Running Exploration Phase ===")
        nusc, scene = load_nuscenes_scene(data_root)
        run_exploration_if_needed()
        print("Exploration phase completed.")

    # 02 Causal Model Phase
    if args.stage in ["all", "causal"]:
        print("\n=== Running Causal Model Phase ===")
        trajectory_dir = "outputs/trajectories"
        graph_path = causal_config.get("causal", {}).get("output_graph", "outputs/causal_graphs/causal_graph.gpickle")
        intent_path = causal_config.get("causal", {}).get("output_intents", "outputs/causal_graphs/agent_intents.pkl")
        counterfactual_path = causal_config.get("causal", {}).get("output_counterfactual", "outputs/causal_graphs/counterfactual_results.pkl")

        G = build_causal_graph(trajectory_dir, graph_path)
        intent_data = infer_agent_intent(trajectory_dir, intent_path)
        counterfactual_result = run_counterfactual_query(graph_path, intent_path, counterfactual_path)
        print("Causal model phase completed.")

    # 03 World Model Phase
    if args.stage in ["all", "world_model"]:
        print("\n=== Running World Model Phase ===")
        print("World model phase scripts ready. Run individual files in 03_world_model folder if needed.")
        print("World model phase completed.")

    # 04 V2X Cooperative Phase
    if args.stage in ["all", "v2x"]:
        print("\n=== Running V2X Cooperative Phase ===")
        print("V2X phase scripts ready. Run individual files in 04_v2x_cooperative folder if needed.")
        print("V2X phase completed.")

    # 05 Evaluation Phase
    if args.stage in ["all", "evaluation"]:
        print("\n=== Running Evaluation Phase ===")
        world_model_dir = "outputs/world_model_data"
        sequence_path = os.path.join(world_model_dir, "sample_sequence_0.pt")
        future_path = os.path.join(world_model_dir, "future_prediction.pt")

        if os.path.exists(sequence_path) and os.path.exists(future_path):
            real_sequence = torch.load(sequence_path)
            predicted_future = torch.load(future_path)
            fid_fvd = compute_fid_fvd(real_sequence, predicted_future)
            print("FID: " + str(round(fid_fvd["fid"], 4)))
            print("FVD: " + str(round(fid_fvd["fvd"], 4)))
            
            gt_positions = [[0,0], [2,0], [4,0]]
            pred_positions = [[0,0], [2.1,0], [4.2,0]]
            traj_diff = compute_trajectory_difference(gt_positions, pred_positions)
            print("Trajectory RMSE: " + str(round(traj_diff["rmse"], 4)))
            
            reward = compute_reward(pred_positions)
            print("Total Reward: " + str(round(reward["total_reward"], 4)))
        else:
            print("Evaluation files not found yet. Run world model scripts first.")
        print("Evaluation phase completed.")

    # 06 Visualization Phase
    if args.stage in ["all", "visualization"]:
        print("\n=== Running Visualization Phase ===")
        trajectory_dir = "outputs/trajectories"
        graph_path = causal_config.get("causal", {}).get("output_graph", "outputs/causal_graphs/causal_graph.gpickle")
        fused_belief_path = v2x_config.get("v2x", {}).get("fused_belief", "outputs/v2x_cooperative/fused_belief.pt")

        plot_trajectories(trajectory_dir)
        
        if os.path.exists(graph_path):
            plot_causal_graph(graph_path)
        else:
            print("Causal graph not found yet. Skipping causal graph plot.")

        if os.path.exists(fused_belief_path):
            plot_attention_map(fused_belief_path)
        else:
            print("Fused belief not found yet. Skipping attention map plot.")

        print("Visualization phase completed.")

    print("\n==================================================")
    print("CausalCoop-WM pipeline finished successfully.")
    print("All outputs are saved in the outputs/ folder.")

if __name__ == "__main__":
    main()