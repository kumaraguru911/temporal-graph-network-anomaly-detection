import pandas as pd
import networkx as nx
from pathlib import Path
import pickle

FLOW_DATA_DIR = Path("data/intermediate")
USER_DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/processed/graphs")


def build_graph_for_window(flow_df, user_df, window_start):
    G = nx.DiGraph()

    window_users = user_df[user_df["window_start"] == window_start]

    for _, row in window_users.iterrows():
        node_id = row["src_ip"]
        features = row.drop(labels=["window_start", "src_ip"]).to_dict()
        G.add_node(node_id, **features)

    window_flows = flow_df[flow_df["window_start"] == window_start]

    for _, row in window_flows.iterrows():
        src = row["src_ip"]
        dst = row["dst_ip"]

        if src not in G or dst not in G:
            continue


        G.add_edge(
            src,
            dst,
            packet_count=row["packet_count"],
            total_bytes=row["total_bytes"],
            avg_packet_size=row["avg_packet_size"],
            flow_duration=row["flow_duration"],
        )

    return G


def run_graph_builder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for flow_file in FLOW_DATA_DIR.glob("flow_*.csv"):
        print(f"[+] Processing {flow_file.name}")

        flow_df = pd.read_csv(flow_file)
        timestamp = flow_file.stem.replace("flow_", "").replace("flows_", "")
        user_file = USER_DATA_DIR / f"user_behavior_{timestamp}.csv"

        if not user_file.exists():
            print(f"[!] User behavior file not found, skipping: {user_file}")
            continue

        user_df = pd.read_csv(user_file)


        for window_start in flow_df["window_start"].unique():
            G = build_graph_for_window(flow_df, user_df, window_start)

            graph_file = (
                OUTPUT_DIR
                / f"graph_{flow_file.stem}_window_{int(window_start)}.gpickle"
            )

            with open(graph_file, "wb") as f:
                pickle.dump(G, f)

            print(f"    [+] Saved graph → {graph_file}")


if __name__ == "__main__":
    run_graph_builder()
