import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.data import Data

from src.models.vgae_sage import GraphSAGEVAE
from src.anomaly.anomaly_scoring import compute_node_anomaly_scores


DEVICE = torch.device("cpu")
DATA_DIR = Path("data/processed/ml_ready")
EPOCHS = 100
LR = 0.001

def load_graph_tensors(x_file, edge_file):
    X = np.load(x_file)

    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    X = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(np.load(edge_file), dtype=torch.long)

    return Data(x=X, edge_index=edge_index)

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

def train_baseline_model(graph_list):

    in_channels = graph_list[0].num_node_features

    model = GraphSAGEVAE(
        in_channels=in_channels,
        hidden_channels=32,
        latent_channels=16
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):

        total_loss = 0

        for data in graph_list:

            model.train()
            optimizer.zero_grad()

            x_recon, adj_logits, mu, logvar = model(data.x, data.edge_index)

            feature_loss = F.mse_loss(x_recon, data.x)

            adj_true = torch.zeros(data.num_nodes, data.num_nodes)
            adj_true[data.edge_index[0], data.edge_index[1]] = 1

            adj_loss = F.binary_cross_entropy_with_logits(
                adj_logits,
                adj_true
            )

            kld = kl_loss(mu, logvar)

            loss = feature_loss + adj_loss + 0.001 * kld
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Avg Loss: {total_loss / len(graph_list):.4f}")

    return model

def inject_heavy_attack(data):

    attacked_data = Data(
        x=data.x.clone(),
        edge_index=data.edge_index.clone()
    )

    attacker = 0
    num_nodes = attacked_data.num_nodes

    attacked_data.x[attacker] *= 5.0

    new_edges = []
    for target in range(num_nodes):
        if target != attacker:
            new_edges.append([attacker, target])

    if len(new_edges) > 0:
        new_edges = torch.tensor(new_edges).t()
        attacked_data.edge_index = torch.cat(
            [attacked_data.edge_index, new_edges],
            dim=1
        )

    return attacked_data

def main():

    x_files = sorted(DATA_DIR.glob("*_X.npy"))
    edge_files = sorted(DATA_DIR.glob("*_edge_index.npy"))

    graph_list = []

    for x_file, edge_file in zip(x_files, edge_files):
        graph_list.append(load_graph_tensors(x_file, edge_file))

    print("Training baseline model on normal windows...\n")
    model = train_baseline_model(graph_list)
    print("\nBaseline training complete.\n")

    print("Baseline Evaluation")

    baseline_embeddings = []
    baseline_drift_values = []

    for i, data in enumerate(graph_list):

        anomaly_scores, embeddings = compute_node_anomaly_scores(model, data)

        print(f"Window {i} baseline anomaly mean:",
              anomaly_scores.mean().item())

        baseline_embeddings.append(embeddings)

    for t in range(1, len(baseline_embeddings)):

        prev = baseline_embeddings[t-1]
        curr = baseline_embeddings[t]

        min_nodes = min(prev.size(0), curr.size(0))

        drift = torch.norm(
            curr[:min_nodes] - prev[:min_nodes],
            dim=1
        )

        baseline_drift_values.append(drift)

    baseline_drift_concat = torch.cat(baseline_drift_values)

    drift_mean = baseline_drift_concat.mean().item()
    drift_std = baseline_drift_concat.std().item()
    drift_threshold = drift_mean + 2 * drift_std

    print("\nBaseline Drift Threshold:", drift_threshold)

    print("\nAttack Evaluation")

    attacked_graph = inject_heavy_attack(graph_list[1])

    anomaly_scores_attack, embeddings_attack = compute_node_anomaly_scores(
        model, attacked_graph
    )

    print("Attack anomaly mean:",
          anomaly_scores_attack.mean().item())

    prev_normal = baseline_embeddings[0]

    min_nodes = min(prev_normal.size(0), embeddings_attack.size(0))

    attack_drift = torch.norm(
        embeddings_attack[:min_nodes] - prev_normal[:min_nodes],
        dim=1
    )

    print("Attack drift mean:",
          attack_drift.mean().item())

    print("Drift threshold:", drift_threshold)

    if attack_drift.mean().item() > drift_threshold:
        print("\n🚨 Temporal Anomaly DETECTED")
    else:
        print("\nNo temporal anomaly detected")


if __name__ == "__main__":
    main()