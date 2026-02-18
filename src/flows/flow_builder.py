import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/intermediate")
WINDOW_SIZE = 30  # seconds

def build_flows(packet_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(
    packet_csv,
    header=None,
    names=["timestamp", "src_ip", "dst_ip", "protocol", "length"],
    on_bad_lines="skip"
    )

    df.columns = ["timestamp", "src_ip", "dst_ip", "protocol", "length"]

    df["timestamp"] = df["timestamp"].astype(float)

    df["window_start"] = df["timestamp"] // WINDOW_SIZE * WINDOW_SIZE

    flows = (
        df.groupby(["window_start","src_ip", "dst_ip", "protocol"]).agg(
            packet_count=("length", "count"),
            total_bytes=("length", "sum"),
            avg_packet_size=("length", "mean"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
        ).reset_index()
    )

    flows["flow_duration"] = flows["last_seen"] - flows["first_seen"]

    flows = flows.drop(columns=["first_seen", "last_seen"])

    return flows
def run_flow_builder():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for packet_file in RAW_DATA_DIR.glob("packets_*.csv"):
        print(f"[+] Processing {packet_file.name}...")
        flows = build_flows(packet_file)
        output_file = OUTPUT_DIR / packet_file.name.replace("packets_", "flow_")
        flows.to_csv(output_file, index=False)
        print(f"[+] Flows saved to {output_file}")

if __name__ == "__main__":
    run_flow_builder()