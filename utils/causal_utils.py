import os
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict


def save_graph(G, path):
    """Save networkx graph using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(G, f)


def load_graph(path):
    """Load networkx graph using pickle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_causal_graph(trajectory_dir, output_path):
    """
    Build a causal graph over agents from trajectory data.
    Nodes = agents, Edges = spatial proximity + interaction.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traj_files = [
        f for f in os.listdir(trajectory_dir)
        if f.endswith('.pkl')
    ]

    if len(traj_files) == 0:
        print("No trajectory files found. Run extraction first.")
        G = nx.DiGraph()
        save_graph(G, output_path)
        return G

    traj_path = os.path.join(trajectory_dir, traj_files[0])
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    G      = nx.DiGraph()
    agents = list(trajectories.keys())

    for inst_token, traj in trajectories.items():
        G.add_node(
            inst_token,
            category=traj[0]['category'],
            frames=len(traj)
        )

    max_distance = 20.0
    print(f"Building causal graph for {len(agents)} agents...")

    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue

            traj_i  = trajectories[agents[i]]
            traj_j  = trajectories[agents[j]]
            min_len = min(len(traj_i), len(traj_j))

            if min_len == 0:
                continue

            distances = []
            for k in range(min_len):
                dx = traj_i[k]['x'] - traj_j[k]['x']
                dy = traj_i[k]['y'] - traj_j[k]['y']
                distances.append(np.sqrt(dx**2 + dy**2))

            avg_dist = np.mean(distances)
            min_dist = np.min(distances)

            if min_dist < max_distance:
                weight = 1.0 / (min_dist + 1e-6)
                G.add_edge(
                    agents[i], agents[j],
                    weight=weight,
                    avg_distance=float(avg_dist),
                    min_distance=float(min_dist)
                )

    save_graph(G, output_path)

    print(f"Causal graph saved : {output_path}")
    print(f"  Nodes            : {G.number_of_nodes()}")
    print(f"  Edges            : {G.number_of_edges()}")
    return G


def infer_agent_intent(trajectory_dir, output_path):
    """
    Infer intent for each agent based on trajectory pattern.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    traj_files = [
        f for f in os.listdir(trajectory_dir)
        if f.endswith('.pkl')
    ]

    if len(traj_files) == 0:
        print("No trajectory files found.")
        return {}

    traj_path = os.path.join(trajectory_dir, traj_files[0])
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)

    intent_data = {}

    for inst_token, traj in trajectories.items():
        if len(traj) < 2:
            intent = 'stationary'
        else:
            dx_total = traj[-1]['x'] - traj[0]['x']
            dy_total = traj[-1]['y'] - traj[0]['y']
            dist     = np.sqrt(dx_total**2 + dy_total**2)

            if dist < 1.0:
                intent = 'stationary'
            elif abs(dy_total) < 2.0:
                intent = 'moving_straight'
            elif dy_total > 2.0:
                intent = 'turning_left'
            elif dy_total < -2.0:
                intent = 'turning_right'
            else:
                intent = 'stopping'

        intent_data[inst_token] = {
            'intent'  : intent,
            'category': traj[0]['category'],
            'frames'  : len(traj)
        }

    with open(output_path, 'wb') as f:
        pickle.dump(intent_data, f)

    print(f"Agent intents saved: {output_path}")

    from collections import Counter
    intents = [v['intent'] for v in intent_data.values()]
    counts  = Counter(intents)
    for intent, count in counts.items():
        print(f"  {intent:<25} {count} agents")

    return intent_data


def run_counterfactual_query(graph_path, intent_path, output_path):
    """
    Run a counterfactual query on the causal graph.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(graph_path):
        print("Causal graph not found. Run build_causal_graph first.")
        return {}

    if not os.path.exists(intent_path):
        print("Intent data not found. Run infer_agent_intent first.")
        return {}

    G = load_graph(graph_path)

    with open(intent_path, 'rb') as f:
        intent_data = pickle.load(f)

    results = {}
    nodes   = list(G.nodes())

    if len(nodes) == 0:
        print("Graph has no nodes.")
        return {}

    query_agent      = nodes[0]
    original_intent  = intent_data.get(
        query_agent, {}
    ).get('intent', 'unknown')

    counterfactual_intents = [
        'moving_straight',
        'turning_left',
        'turning_right',
        'stopping'
    ]

    print(f"\nQuery agent    : {query_agent[:12]}...")
    print(f"Original intent: {original_intent}")
    print()

    for cf_intent in counterfactual_intents:
        if cf_intent == original_intent:
            continue

        neighbors   = list(G.successors(query_agent))
        affected    = len(neighbors)
        risk_change = round(affected * 0.15, 3)
        outcome     = 'increased risk' if risk_change > 0.3 else 'safe'

        results[cf_intent] = {
            'query_agent'    : query_agent,
            'original_intent': original_intent,
            'counterfactual' : cf_intent,
            'affected_agents': affected,
            'risk_change'    : risk_change,
            'outcome'        : outcome
        }

        print(
            f"  '{original_intent}' → '{cf_intent}' : "
            f"{affected} agents affected, "
            f"risk={risk_change}, {outcome}"
        )

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\nCounterfactual results saved: {output_path}")
    return results