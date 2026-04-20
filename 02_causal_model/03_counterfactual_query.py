import os
import pickle
import networkx as nx

GRAPH_PATH  = 'outputs/causal_graphs/causal_graph.gpickle'
INTENT_PATH = 'outputs/causal_graphs/agent_intents.pkl'
OUTPUT_PATH = 'outputs/causal_graphs/counterfactual_results.pkl'

if not os.path.exists(GRAPH_PATH):
    print("Causal graph not found. Run 01_build_causal_graph.py first.")
    exit()

if not os.path.exists(INTENT_PATH):
    print("Intent data not found. Run 02_agent_intent_inference.py first.")
    exit()

import pickle as pkl
with open(GRAPH_PATH, 'rb') as f:
    G = pkl.load(f)

with open(INTENT_PATH, 'rb') as f:
    intent_data = pickle.load(f)

print(f"Graph loaded  : {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Agents loaded : {len(intent_data)}")
print()

nodes = list(G.nodes())

if len(nodes) == 0:
    print("Graph has no nodes.")
    exit()

query_agent      = nodes[0]
original_intent  = intent_data.get(query_agent, {}).get('intent', 'unknown')
original_category= intent_data.get(query_agent, {}).get('category', 'unknown')

print(f"Query agent   : {query_agent[:12]}...")
print(f"Category      : {original_category}")
print(f"Original intent: {original_intent}")
print()

counterfactual_intents = [
    'moving_straight',
    'turning_left',
    'turning_right',
    'stopping',
    'stationary'
]

results = {}

print('COUNTERFACTUAL ANALYSIS:')
print('=' * 60)

for cf_intent in counterfactual_intents:
    if cf_intent == original_intent:
        continue

    neighbors      = list(G.successors(query_agent))
    affected       = len(neighbors)
    risk_change    = round(affected * 0.15, 3)
    outcome        = 'increased risk' if risk_change > 0.3 else 'safe'

    results[cf_intent] = {
        'query_agent'    : query_agent,
        'original_intent': original_intent,
        'counterfactual' : cf_intent,
        'affected_agents': affected,
        'risk_change'    : risk_change,
        'outcome'        : outcome
    }

    print(f"  If intent changes: '{original_intent}' → '{cf_intent}'")
    print(f"    Affected agents : {affected}")
    print(f"    Risk change     : {risk_change}")
    print(f"    Outcome         : {outcome}")
    print()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump(results, f)

print(f"Counterfactual results saved: {OUTPUT_PATH}")