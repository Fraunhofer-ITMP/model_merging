from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import get_xy, load_tsv
from model import SolubilityMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test saved MLP model on a TSV dataset.")
    parser.add_argument("--model-path", type=Path, default=Path("output/solubility_mlp.pt"))
    parser.add_argument("--data-path", type=Path, default=Path("input/solub_df1_holdback.tsv"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-metrics", type=Path, default=Path("output/test_metrics.json"))
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            labels_all.append(yb.cpu().numpy())

    probs = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    preds = (probs >= 0.5).astype(np.float32)

    result = {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    try:
        result["roc_auc"] = roc_auc_score(labels, probs)
    except ValueError:
        result["roc_auc"] = float("nan")
    return result


def main() -> None:
    args = parse_args()
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)

    # The checkpoint is created locally by train.py and includes scaler arrays/metadata.
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    feature_columns = checkpoint["feature_columns"]
    label_column = checkpoint["label_column"]
    hidden_dims = checkpoint["hidden_dims"]
    dropout = checkpoint["dropout"]
    
    model = SolubilityMLP(
        input_dim=len(feature_columns),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    df = load_tsv(args.data_path)
    x, y, _ = get_xy(df, feature_columns, label_column)

    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    pos_count = float((y == 1).sum())
    neg_count = float((y == 0).sum())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    metrics = evaluate(model, loader, criterion, torch.device("cpu"))
    with open(args.output_metrics, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

    print("Test metrics:", {k: round(float(v), 5) for k, v in metrics.items()})
    print(f"Saved metrics to: {args.output_metrics}")


if __name__ == "__main__":
    main()
