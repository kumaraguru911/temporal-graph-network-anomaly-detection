
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from capture.live_capture import start_live_capture
from flows.flow_builder import run_flow_builder
from graphs.user_behavior_builder import run_user_behavior_builder
from graphs.graph_builder import run_graph_builder
from graphs.graph_to_ml import run_conversion
from training.train_vgae import main as train_main

def main():
    print("🚀 Starting Temporal Graph Network Anomaly Detection Pipeline\n")

    # Step 1: Capture live packet data
    print("Step 1: Capturing live packet data...")
    try:
        start_live_capture()
        print("✅ Packet capture complete.\n")
    except Exception as e:
        print(f"❌ Error in packet capture: {e}")
        print("Note: Packet capture may require sudo privileges.")
        return

    # Step 2: Build flows from packets
    print("Step 2: Building flows from packet data...")
    try:
        run_flow_builder()
        print("✅ Flow building complete.\n")
    except Exception as e:
        print(f"❌ Error in flow building: {e}")
        return

    # Step 3: Build user behavior profiles
    print("Step 3: Building user behavior profiles...")
    try:
        run_user_behavior_builder()
        print("✅ User behavior building complete.\n")
    except Exception as e:
        print(f"❌ Error in user behavior building: {e}")
        return

    # Step 4: Build graphs
    print("Step 4: Building graphs...")
    try:
        run_graph_builder()
        print("✅ Graph building complete.\n")
    except Exception as e:
        print(f"❌ Error in graph building: {e}")
        return

    # Step 5: Convert graphs to ML-ready tensors
    print("Step 5: Converting graphs to ML tensors...")
    try:
        run_conversion()
        print("✅ ML conversion complete.\n")
    except Exception as e:
        print(f"❌ Error in ML conversion: {e}")
        return

    # Step 6: Train the model and evaluate
    print("Step 6: Training model and evaluating anomalies...")
    try:
        train_main()
        print("✅ Training and evaluation complete.\n")
    except Exception as e:
        print(f"❌ Error in training: {e}")
        return

    print("🎉 Pipeline execution finished successfully!")

if __name__ == "__main__":
    main()