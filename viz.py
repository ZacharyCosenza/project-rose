"""
viz.py — visualizations for project-rose.

Subcommands:
    python viz.py kg              Render knowledge graph → kg_graph.html
    python viz.py kg --top 50     Limit to top 50 entities by degree
    python viz.py kg --output my.html
"""

import argparse
import json
from pathlib import Path

KG_PATH = Path("kg.json")


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------

def cmd_kg(top: int | None, output: Path) -> None:
    import networkx as nx
    from pyvis.network import Network

    if not KG_PATH.exists():
        print(f"Error: {KG_PATH} not found. Run 'python training.py dedupe' first.")
        return

    triples = json.loads(KG_PATH.read_text(encoding="utf-8"))

    # Deduplicate to unique canonical triples
    seen: set[tuple] = set()
    unique_triples = []
    for t in triples:
        key = (t["canonical_subject"], t["canonical_predicate"], t["canonical_object"])
        if key not in seen:
            seen.add(key)
            unique_triples.append(t)

    # Build directed graph
    G = nx.DiGraph()
    for t in unique_triples:
        s, p, o = t["canonical_subject"], t["canonical_predicate"], t["canonical_object"]
        G.add_node(s)
        G.add_node(o)
        # Parallel edges (same s→o, different predicate) are merged into one edge
        # with a combined label — keeps the graph readable
        if G.has_edge(s, o):
            existing = G[s][o]["label"]
            if p not in existing.split(" / "):
                G[s][o]["label"] = existing + " / " + p
        else:
            G.add_edge(s, o, label=p)

    if top:
        top_nodes = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:top]
        G = G.subgraph(set(top_nodes)).copy()

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    net = Network(
        height="920px",
        width="100%",
        directed=True,
        bgcolor="#0f0f1a",
        font_color="#e0e0e0",
    )

    # Add nodes sized and colored by degree
    max_degree = max((G.degree(n) for n in G.nodes()), default=1)
    for node in G.nodes():
        degree = G.degree(node)
        size = 12 + (degree / max_degree) * 40
        net.add_node(
            node,
            label=node,
            size=size,
            title=f"{node}<br>{degree} connection{'s' if degree != 1 else ''}",
            color={
                "background": "#4a9eff",
                "border":     "#1a6ecf",
                "highlight":  {"background": "#7bbfff", "border": "#4a9eff"},
            },
            font={"size": 12},
        )

    for s, o, data in G.edges(data=True):
        label = data.get("label", "")
        net.add_edge(
            s, o,
            label=label,
            title=label,
            color={"color": "#555577", "highlight": "#9999cc"},
            arrows="to",
            font={"size": 9, "color": "#aaaacc", "strokeWidth": 0},
            smooth={"type": "continuous"},
        )

    net.set_options("""
    {
      "physics": {
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08,
          "damping": 0.6
        },
        "stabilization": { "iterations": 150, "updateInterval": 25 }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    net.write_html(str(output))
    print(f"Saved → {output}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="viz.py",
        description="project-rose visualizations",
    )
    subparsers = parser.add_subparsers(dest="command")

    kg_parser = subparsers.add_parser("kg", help="Render knowledge graph → HTML")
    kg_parser.add_argument("--top",    type=int,  default=None,                  help="Limit to top N entities by degree")
    kg_parser.add_argument("--output", type=Path, default=Path("kg_graph.html"), help="Output file (default: kg_graph.html)")

    args = parser.parse_args()

    if args.command == "kg":
        cmd_kg(top=args.top, output=args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
