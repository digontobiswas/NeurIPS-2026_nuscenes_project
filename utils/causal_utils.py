import os
import pickle
import networkx as nx
import numpy as np

def build_causal_graph(trajectory_dir, output_path=None):
    print("Building causal graph from trajectories")
    trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
    all_trajectories = {}
    
    for tf in trajectory_files:
        with open(os.path.join(trajectory_dir, tf), "rb") as f:
            traj = pickle.load(f)
        instance_token = tf.replace("traj_", "").replace(".pkl", "")
        all_trajectories[instance_token] = traj
    
    print(f"Loaded {len(all_trajectories)} agent trajectories")

    G = nx.DiGraph()
    for instance_token, traj in all_trajectories.items():
        G.add_node(instance_token, category=traj[0]["category"], frames=len(traj))

    print(f"Added {G.number_of_nodes()} nodes to the graph")

    # Add edges
    edge_count = 0
    for i1, traj1 in all_trajectories.items():
        for i2, traj2 in all_trajectories.items():
            if i1 == i2:
                continue
            for t in range(min(len(traj1), len(traj2))):
                pos1 = np.array(traj1[t]["translation"][:2])
                pos2 = np.array(traj2[t]["translation"][:2])
                dist = np.linalg.norm(pos1 - pos2)
                if dist < 15.0:
                    vel1 = np.array(traj1[t].get("velocity", [0,0]))
                    vel2 = np.array(traj2[t].get("velocity", [0,0]))
                    if np.dot(vel1, vel2) < 0:
                        G.add_edge(i1, i2, weight=1.0 / (dist + 1e-6), time=t)
                        edge_count += 1
                        break

    print(f"Added {edge_count} causal edges")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(G, f)
        print("Causal graph saved to " + output_path)
    
    return G

def infer_agent_intent(trajectory_dir, output_path=None):
    print("Inferring agent intents")
    trajectory_files = [f for f in os.listdir(trajectory_dir) if f.endswith(".pkl")]
    intent_data = {}
    
    for tf in trajectory_files:
        with open(os.path.join(trajectory_dir, tf), "rb") as f:
            traj = pickle.load(f)
        instance_token = tf.replace("traj_", "").replace(".pkl", "")
        
        intents = []
        for i in range(1, len(traj)):
            prev_pos = np.array(traj[i-1]["translation"][:2])
            curr_pos = np.array(traj[i]["translation"][:2])
            prev_vel = np.linalg.norm(np.array(traj[i-1].get("velocity", [0,0])))
            curr_vel = np.linalg.norm(np.array(traj[i].get("velocity", [0,0])))
            
            speed_change = curr_vel - prev_vel
            direction_change = np.arctan2(curr_pos[1] - prev_pos[1], curr_pos[0] - prev_pos[0])
            
            if speed_change < -0.5:
                intent = "braking"
            elif speed_change > 0.5:
                intent = "accelerating"
            elif abs(direction_change) > 0.3:
                intent = "turning"
            else:
                intent = "cruising"
            intents.append(intent)
        
        intent_data[instance_token] = {
            "category": traj[0]["category"],
            "intents": intents,
            "most_common_intent": max(set(intents), key=intents.count) if intents else "unknown"
        }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(intent_data, f)
        print("Agent intents saved to " + output_path)
    return intent_data

def run_counterfactual_query(graph_path, intent_path, output_path=None):
    print("Running counterfactual query")
    if not os.path.exists(graph_path) or not os.path.exists(intent_path):
        print("Graph or intent files not found")
        return None
    
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    if G.number_of_nodes() == 0:
        print("WARNING: Graph has no nodes. Skipping counterfactual query.")
        return None
    
    with open(intent_path, "rb") as f:
        intent_data = pickle.load(f)
    
    ego_token = list(G.nodes())[0]
    print(f"Using agent {ego_token[:8]} as ego for counterfactual")
    
    targets = list(G.successors(ego_token))
    
    original_influence = len(nx.shortest_path(G, ego_token, targets[0])) if targets else 0
    
    G_counter = G.copy()
    for u, v, data in list(G_counter.edges(data=True)):
        if u == ego_token and intent_data.get(u, {}).get("most_common_intent") == "braking":
            G_counter.remove_edge(u, v)
    
    new_influence = len(nx.shortest_path(G_counter, ego_token, targets[0])) if targets else 0
    
    result = {
        "ego": ego_token,
        "original_influence": original_influence,
        "counterfactual_influence": new_influence,
        "change": new_influence - original_influence
    }
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
        print("Counterfactual results saved to " + output_path)
    return result