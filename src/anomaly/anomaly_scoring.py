import torch
import torch.nn.functional as F


def compute_node_anomaly_scores(model, data):
    model.eval()

    with torch.no_grad():
        x_recon, adj_recon_logits, mu, logvar = model(data.x, data.edge_index)

        feature_error = torch.mean(
            (x_recon - data.x) ** 2,
            dim=1
        )

        edge_logits = adj_recon_logits[
            data.edge_index[0],
            data.edge_index[1]
        ]

        edge_labels = torch.ones_like(edge_logits)

        edge_loss = F.binary_cross_entropy_with_logits(
            edge_logits,
            edge_labels,
            reduction="none"
        )

        structural_error = torch.zeros(data.num_nodes)

        for i, src in enumerate(data.edge_index[0]):
            structural_error[src] += edge_loss[i]

        degree = torch.bincount(
            data.edge_index[0],
            minlength=data.num_nodes
        ).float()

        structural_error = structural_error / (degree + 1e-6)

        anomaly_score = feature_error + structural_error

    return anomaly_score, mu