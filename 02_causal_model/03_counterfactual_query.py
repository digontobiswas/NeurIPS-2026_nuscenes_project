import os
import pickle
import networkx as nx
import numpy as np

print("CausalCoop-WM Counterfactual Query")
print("==================================================")

graph_path = "outputs/causal_graphs/causal_graph.gpickle"
intent_path = "outputs/causal_graphs/agent_intents.pkl"

if not os.path.exists(graph_path) or not os.path.exists(intent_path):
    print("Causal graph or intents not found. Run previous scripts first.")
else:
    G = nx.read_gpickle(graph_path)
    with open(intent_path, "rb") as f:
        intent_data = pickle.load(f)
    
    # Example counterfactual: "What if ego agent did not brake?"
    ego_token = list(G.nodes())[0]  # first agent as ego
    print("Running counterfactual on ego agent: " + ego_token[:8])
    
    # Original path length to a target
    targets = list(G.successors(ego_token))
    if targets:
        original_influence = len(nx.shortest_path(G, ego_token, targets[0]))
    else:
        original_influence = 0
    
    # Simulate intervention: remove braking edge
    G_counter = G.copy()
    for u, v, data in list(G_counter.edges(data=True)):
        if u == ego_token and intent_data.get(u, {}).get("most_common_intent") == "braking":
            G_counter.remove_edge(u, v)
    
    # New influence
    if targets:
        try:
            new_influence = len(nx.shortest_path(G_counter, ego_token, targets[0]))
        except:
            new_influence = 0
    else:
        new_influence = 0
    
    print("Original causal influence length: " + str(original_influence))
    print("Counterfactual (no brake) influence length: " + str(new_influence))
    print("Change in influence: " + str(new_influence - original_influence))
    
    with open("outputs/causal_graphs/counterfactual_results.pkl", "wb") as f:
        pickle.dump({
            "ego": ego_token,
            "original_influence": original_influence,
            "counterfactual_influence": new_influence
        }, f)
    
    print("Counterfactual query completed and saved.")