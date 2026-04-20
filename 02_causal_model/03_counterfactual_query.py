import os
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GRAPH_PATH  = 'outputs/causal_graphs/causal_graph_moving.gpickle'
INTENT_PATH = 'outputs/causal_graphs/agent_intents.pkl'
OUTPUT_PATH = 'outputs/causal_graphs/counterfactual_results.pkl'
FIGURES     = 'outputs/figures'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ── Load graph and intents ────────────────────────────────────
if not os.path.exists(GRAPH_PATH):
    print("Moving graph not found. Run 01_build_causal_graph.py first.")
    exit()

if not os.path.exists(INTENT_PATH):
    print("Intent data not found. Run 02_agent_intent_inference.py first.")
    exit()

with open(GRAPH_PATH, 'rb') as f:
    G = pickle.load(f)

with open(INTENT_PATH, 'rb') as f:
    intent_data = pickle.load(f)

print(f"Graph loaded   : {G.number_of_nodes()} nodes, "
      f"{G.number_of_edges()} edges")
print(f"Agents loaded  : {len(intent_data)}")
print()

# ── Select moving agents only ─────────────────────────────────
moving_agents = {
    token: data
    for token, data in intent_data.items()
    if data.get('is_moving', False)
}

print(f"Moving agents  : {len(moving_agents)}")
print()


def compute_risk_score(G, agent_token, intent_data):
    """
    Compute risk score for an agent based on:
    - Number of neighbors influenced
    - Average edge weight to neighbors
    - Speed of influenced neighbors
    """
    if agent_token not in G:
        return 0.0

    neighbors = list(G.successors(agent_token))
    if len(neighbors) == 0:
        return 0.0

    total_weight = sum(
        G[agent_token][n].get('weight', 0)
        for n in neighbors
    )
    avg_weight   = total_weight / len(neighbors)
    neighbor_speeds = []

    for n in neighbors:
        stats = intent_data.get(n, {}).get('stats', {})
        speed = stats.get('avg_speed', 0)
        neighbor_speeds.append(speed)

    avg_neighbor_speed = np.mean(neighbor_speeds) if neighbor_speeds else 0

    risk = (
        0.4 * len(neighbors) / max(G.number_of_nodes(), 1) +
        0.4 * avg_weight +
        0.2 * min(avg_neighbor_speed / 10.0, 1.0)
    )
    return float(np.clip(risk, 0, 1))


def run_counterfactual(G, agent_token, original_intent,
                       cf_intent, intent_data):
    """
    Simulate what happens when one agent's intent changes.
    Returns a dict of predicted effects on neighbors.
    """
    neighbors = list(G.successors(agent_token))
    effects   = {}

    intent_speed_map = {
        'moving_straight': 5.0,
        'turning_left'   : 3.0,
        'turning_right'  : 3.0,
        'stopping'       : 0.5,
        'stationary'     : 0.0,
        'moving_fast'    : 10.0
    }

    original_speed = intent_speed_map.get(original_intent, 2.0)
    cf_speed       = intent_speed_map.get(cf_intent, 2.0)
    speed_change   = cf_speed - original_speed

    for neighbor in neighbors:
        edge_weight  = G[agent_token][neighbor].get('weight', 0)
        n_intent     = intent_data.get(neighbor, {}).get('intent', 'unknown')
        n_speed      = intent_data.get(neighbor, {}).get(
            'stats', {}
        ).get('avg_speed', 0)

        predicted_speed_change = speed_change * edge_weight
        new_speed              = max(0, n_speed + predicted_speed_change)

        risk_increase = abs(predicted_speed_change) * edge_weight
        collision_risk = min(1.0, risk_increase * 2)

        effects[neighbor] = {
            'original_speed'      : n_speed,
            'predicted_new_speed' : new_speed,
            'speed_change'        : predicted_speed_change,
            'edge_weight'         : edge_weight,
            'neighbor_intent'     : n_intent,
            'collision_risk'      : collision_risk
        }

    return effects


# ── Run counterfactual for top 3 most influential agents ──────
nodes_in_graph = [
    n for n in moving_agents.keys()
    if n in G
]

if len(nodes_in_graph) == 0:
    print("No moving agents found in graph.")
    exit()

sorted_agents = sorted(
    nodes_in_graph,
    key=lambda n: G.out_degree(n),
    reverse=True
)[:3]

all_results    = {}
cf_intents_map = {
    'moving_straight': ['turning_left', 'stopping'],
    'turning_left'   : ['moving_straight', 'stopping'],
    'turning_right'  : ['moving_straight', 'stopping'],
    'stopping'       : ['moving_straight', 'turning_left'],
    'stationary'     : ['moving_straight'],
    'moving_fast'    : ['stopping', 'moving_straight']
}

print('=' * 70)
print('COUNTERFACTUAL ANALYSIS — What if an agent changes intent?')
print('=' * 70)

for agent_token in sorted_agents:
    agent_data      = moving_agents.get(agent_token, {})
    original_intent = agent_data.get('intent', 'unknown')
    category        = agent_data.get('category', 'unknown').split('.')[-1]
    original_risk   = compute_risk_score(G, agent_token, intent_data)

    print(f"\nAgent    : {agent_token[:16]}...")
    print(f"Category : {category}")
    print(f"Intent   : {original_intent}")
    print(f"Out-degree: {G.out_degree(agent_token)} neighbors")
    print(f"Base risk : {original_risk:.4f}")
    print()

    cf_list = cf_intents_map.get(original_intent, ['moving_straight'])
    agent_results = {}

    for cf_intent in cf_list:
        effects = run_counterfactual(
            G, agent_token,
            original_intent, cf_intent,
            intent_data
        )

        avg_collision = np.mean([
            e['collision_risk'] for e in effects.values()
        ]) if effects else 0

        max_collision = max([
            e['collision_risk'] for e in effects.values()
        ], default=0)

        high_risk_count = sum(
            1 for e in effects.values()
            if e['collision_risk'] > 0.3
        )

        print(f"  Counterfactual: '{original_intent}' → '{cf_intent}'")
        print(f"    Neighbors affected : {len(effects)}")
        print(f"    Avg collision risk : {avg_collision:.4f}")
        print(f"    Max collision risk : {max_collision:.4f}")
        print(f"    High risk agents   : {high_risk_count}")
        print(f"    Safety outcome     : "
              f"{'DANGEROUS' if avg_collision > 0.3 else 'SAFE'}")
        print()

        agent_results[cf_intent] = {
            'effects'          : effects,
            'avg_collision'    : avg_collision,
            'max_collision'    : max_collision,
            'high_risk_count'  : high_risk_count,
            'original_intent'  : original_intent,
            'counterfactual'   : cf_intent
        }

    all_results[agent_token] = agent_results

# ── Save results ──────────────────────────────────────────────
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(all_results, f)

print(f"Results saved: {OUTPUT_PATH}")

# ── Visualize counterfactual risk ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle(
    'Counterfactual Analysis — PS-1 Causal Reasoning\n'
    'What if an agent changes intent?',
    fontsize=13
)

# Plot 1 — Risk comparison bar chart
agent_labels = []
original_risks = []
cf_risks       = []
cf_labels_list = []

for agent_token, results in all_results.items():
    cat = moving_agents.get(
        agent_token, {}
    ).get('category', '?').split('.')[-1]
    orig_intent = moving_agents.get(
        agent_token, {}
    ).get('intent', '?')
    orig_risk   = compute_risk_score(G, agent_token, intent_data)

    for cf_intent, result in results.items():
        agent_labels.append(f"{cat}\n{orig_intent[:6]}→{cf_intent[:6]}")
        original_risks.append(orig_risk)
        cf_risks.append(result['avg_collision'])
        cf_labels_list.append(cf_intent)

x     = np.arange(len(agent_labels))
width = 0.35

bars1 = axes[0].bar(
    x - width/2, original_risks,
    width, label='Original risk',
    color='steelblue', alpha=0.8
)
bars2 = axes[0].bar(
    x + width/2, cf_risks,
    width, label='Counterfactual risk',
    color='coral', alpha=0.8
)

axes[0].set_title('Risk: Original vs Counterfactual Intent')
axes[0].set_xticks(x)
axes[0].set_xticklabels(agent_labels, fontsize=8)
axes[0].set_ylabel('Risk score')
axes[0].legend()
axes[0].axhline(y=0.3, color='red', linestyle='--',
                alpha=0.5, label='Risk threshold')
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2 — Neighbor collision risk heatmap
if all_results:
    first_agent = list(all_results.keys())[0]
    first_cf    = list(all_results[first_agent].keys())[0]
    effects     = all_results[first_agent][first_cf]['effects']

    neighbor_tokens = list(effects.keys())[:15]
    orig_speeds     = [effects[n]['original_speed']
                       for n in neighbor_tokens]
    new_speeds      = [effects[n]['predicted_new_speed']
                       for n in neighbor_tokens]
    col_risks       = [effects[n]['collision_risk']
                       for n in neighbor_tokens]

    x2 = np.arange(len(neighbor_tokens))
    axes[1].bar(x2, col_risks,
                color=['red' if r > 0.3 else 'green'
                       for r in col_risks],
                alpha=0.8)
    axes[1].axhline(y=0.3, color='red', linestyle='--',
                    label='Risk threshold')
    axes[1].set_title(
        f"Neighbor Collision Risk\n"
        f"Agent: {first_agent[:8]}... "
        f"CF: {first_cf}"
    )
    axes[1].set_xlabel('Neighbor index')
    axes[1].set_ylabel('Collision risk')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    red_patch   = mpatches.Patch(color='red',   label='High risk >0.3')
    green_patch = mpatches.Patch(color='green', label='Safe ≤0.3')
    axes[1].legend(handles=[red_patch, green_patch])

plt.tight_layout()
out_path = os.path.join(FIGURES, 'counterfactual_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Plot saved: {out_path}")
plt.show()
plt.close()