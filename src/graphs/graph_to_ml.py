import pickle
import numpy as np
from pathlib import Path

GRAPH_DIR = Path("data/processed/graphs")
OUTPUT_DIR = Path("data/processed/ml_ready")


def convert_graph(graph_file: Path):
    with open(graph_file, "rb") as f:
        G = pickle.load(f)

    node_list = list(G.nodes())
    node_to_id = {node: i for i, node in enumerate(node_list)}

    features = []
    for node in node_list:
        node_attrs = G.nodes[node]
        features.append(list(node_attrs.values()))

    X = np.array(features, dtype=float)

    edge_index = [[], []]
    for src, dst in G.edges():
        edge_index[0].append(node_to_id[src])
        edge_index[1].append(node_to_id[dst])

    edge_index = np.array(edge_index, dtype=int)

    return X, edge_index, node_to_id


def run_conversion():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for graph_file in GRAPH_DIR.glob("*.gpickle"):
        print(f"[+] Converting {graph_file.name}")

        X, edge_index, node_map = convert_graph(graph_file)

        base = graph_file.stem
        np.save(OUTPUT_DIR / f"{base}_X.npy", X)
        np.save(OUTPUT_DIR / f"{base}_edge_index.npy", edge_index)

        print(f"    Nodes: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"    Edges: {edge_index.shape[1]}")


if __name__ == "__main__":
    run_conversion()
