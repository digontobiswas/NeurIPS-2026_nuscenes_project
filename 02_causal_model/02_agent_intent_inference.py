import os
import pickle
import numpy as np

print("CausalCoop-WM Agent Intent Inference")
print("==================================================")

trajectory_dir = "outputs/trajectories"
output_dir = "outputs/causal_graphs"
os.makedirs(output_dir, exist_ok=True)

# Load all trajectories
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

with open(os.path.join(output_dir, "agent_intents.pkl"), "wb") as f:
    pickle.dump(intent_data, f)

print("Intent inference completed for " + str(len(intent_data)) + " agents")
for token, data in list(intent_data.items())[:5]:
    print("Agent " + token[:8] + " most common intent: " + data["most_common_intent"])
print("Intents saved to outputs/causal_graphs/agent_intents.pkl")