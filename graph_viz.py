"""
graph_viz.py
Step 5: Visualization.

Builds a NetworkX graph of victims/accounts/devices/phones and renders it
(matplotlib figure for reports, plus a Streamlit-friendly PyVis/plotly
option for the interactive dashboard).
"""

import networkx as nx
import matplotlib.pyplot as plt

RISK_COLORS = {"High": "#e63946", "Medium": "#f4a261", "Low": "#2a9d8f"}
TYPE_SHAPES = {
    "account": "s", "upi": "o", "phone": "^", "imei": "d", "ip": "P", "unknown": "o",
}


def build_graph(edges, entity_risk=None):
    """
    edges: DataFrame from correlation.build_entity_edges()
    entity_risk: optional DataFrame from risk_model.score_entities() with
                 columns [entity, risk_level, risk_score]
    Returns a networkx.Graph.
    """
    G = nx.Graph()
    risk_lookup = {}
    if entity_risk is not None:
        if isinstance(entity_risk, dict):
            risk_lookup = entity_risk
        else:
            risk_lookup = entity_risk.set_index("entity")[["risk_level", "risk_score"]].to_dict("index")

    for _, e in edges.iterrows():
        a, b = e["entity_a"], e["entity_b"]
        if not a or not b:
            continue
        G.add_node(a, entity_type=e.get("type_a", "unknown"),
                    **({"risk_level": risk_lookup[a]["risk_level"],
                        "risk_score": risk_lookup[a]["risk_score"]} if a in risk_lookup else {}))
        G.add_node(b, entity_type=e.get("type_b", "unknown"),
                    **({"risk_level": risk_lookup[b]["risk_level"],
                        "risk_score": risk_lookup[b]["risk_score"]} if b in risk_lookup else {}))
        G.add_edge(a, b, relation=e["relation"], amount=e.get("amount"))

    return G


def render_static_graph(G: nx.Graph, output_path: str = "output/network_graph.png",
                         title: str = "Fraud Network Graph"):
    """Render the graph to a PNG (used in the PDF report)."""
    if G.number_of_nodes() == 0:
        return None

    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=0.6)

    node_colors = [
        RISK_COLORS.get(G.nodes[n].get("risk_level", "Low"), "#adb5bd")
        for n in G.nodes
    ]
    node_sizes = [
        300 + 20 * G.nodes[n].get("risk_score", 0) for n in G.nodes
    ]

    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def top_suspicious_subgraph(G: nx.Graph, entity_risk, top_n: int = 15) -> nx.Graph:
    """Return the induced subgraph around the top-N riskiest entities and their neighbors."""
    top_entities = entity_risk.head(top_n)["entity"].tolist()
    nodes = set(top_entities)
    for n in top_entities:
        if n in G:
            nodes.update(G.neighbors(n))
    return G.subgraph(nodes).copy()
