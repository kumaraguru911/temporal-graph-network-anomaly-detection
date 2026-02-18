import subprocess
import datetime
import os

INTERFACE = "any"
CAPTURE_DURATION = 30  # seconds
OUTPUT_DIR = "data/raw"

FIELDS = [
    "frame.time_epoch",
    "ip.src",
    "ip.dst",
    "ip.proto",
    "frame.len"
]

def start_live_capture():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/packets_{timestamp}.csv"

    tshark_cmd = [
        "tshark",
        "-i", INTERFACE,
        "-T", "fields"
    ]

    for field in FIELDS:
        tshark_cmd.extend(["-e", field])

    tshark_cmd.extend([
        "-E", "separator=,",
        "-a", f"duration:{CAPTURE_DURATION}"
    ])

    print("[+] Starting live packet capture...")
    print(f"[+] Output file: {output_file}")

    with open(output_file, "w") as f:
        subprocess.run(tshark_cmd, stdout=f, stderr=subprocess.DEVNULL)

    print("[+] Capture complete.")

if __name__ == "__main__":
    start_live_capture()