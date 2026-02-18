import pandas as pd
from pathlib import Path

FLOW_DATA_DIR = Path("data/intermediate")
OUTPUT_DIR = Path("data/processed")

def build_user_behavior(flow_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(flow_csv)

    user_behavior = (
        df.groupby(["window_start", "src_ip"]).agg(
            total_packets=("packet_count", "sum"),
            total_bytes=("total_bytes", "sum"),
            avg_packet_size=("avg_packet_size", "mean"),
            unique_destinations=("dst_ip", "nunique"),
            unique_protocols=("protocol", "nunique"),
            active_flow_count=("dst_ip", "count"),
        ).reset_index()
    )
    
    return user_behavior

def run_user_behavior_builder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for flow_file in FLOW_DATA_DIR.glob("flow_*.csv"):
        print(f"[+] Processing {flow_file.name}...")

        user_behavior = build_user_behavior(flow_file)

        output_file = OUTPUT_DIR / flow_file.name.replace("flow_", "user_behavior_")
        user_behavior.to_csv(output_file, index=False)
        print(f"[+] User behavior saved to {output_file}")

if __name__ == "__main__":
    run_user_behavior_builder()