from __future__ import annotations

# import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import get_xy, load_tsv
from features import FEATURE_COLUMNS, LABEL_COLUMN
from model import SolubilityMLP


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    # Shared evaluation routine for validation and holdback datasets.
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in dataloader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = yb.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    preds_np = (probs_np >= 0.5).astype(np.float32)

    metrics = {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(labels_np, preds_np),
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }
    try:
        # AUC needs both classes present in labels.
        metrics["roc_auc"] = roc_auc_score(labels_np, probs_np)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP for binary solubility classification.")
    parser.add_argument("--train-path", type=Path, required=True) ## E.g. "model_1/input/solub_df1_train.tsv"
    parser.add_argument("--holdback-path", type=Path, required=True)  # "model_1/input/solub_df1_holdback.tsv"
    parser.add_argument("--output-dir", type=Path, required=True)  # E.g. "model_1/output"

    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0xc0ffee)  # Arbitrary hex seed for fun
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tabular data, extract selected features/label, then split and scale.
    train_df = load_tsv(args.train_path)
    x, y, label_state = get_xy(train_df, FEATURE_COLUMNS, LABEL_COLUMN)
    # x_train, x_val, y_train, y_val = split_train_val(x, y, args.val_size, args.seed)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=args.val_size, random_state=args.seed, stratify=y)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SolubilityMLP(
        input_dim=len(FEATURE_COLUMNS),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    ).to(device)

    # Up- or down-weight positive samples when the training split is imbalanced.
    # In case of the solubility dataset, positives are the majority class, 
    # so this will down-weight them to help the model learn from the minority negative class.
    pos_count = float((y_train == 1).sum())
    neg_count = float((y_train == 0).sum())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_score = -np.inf
    bad_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        history.append(epoch_log)

        val_score = val_metrics["roc_auc"]
        # Fall back to F1 when AUC is undefined.
        if np.isnan(val_score):
            val_score = val_metrics["f1"]

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_score > best_score:
            # Keep full state needed for reproducible inference.
            best_score = val_score
            bad_epochs = 0
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_columns": FEATURE_COLUMNS,
                "label_column": LABEL_COLUMN,
                "label_mapping": label_state.mapping,
                "hidden_dims": args.hidden_dims,
                "dropout": args.dropout,
                "seed": args.seed,
            }
        else:
            bad_epochs += 1
            # Stop when validation score has not improved for `patience` epochs.
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience}).")
                break

    if best_state is None:
        raise RuntimeError("Training ended without a valid model state.")

    model_path = args.output_dir / "solubility_mlp.pt"
    torch.save(best_state, model_path)
    print(f"Saved best model to: {model_path}")

    history_path = args.output_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {history_path}")

    # Evaluate best checkpoint on holdback using train-fitted scaling stats.
    holdback_df = load_tsv(args.holdback_path)
    x_holdback, y_holdback, _ = get_xy(holdback_df, FEATURE_COLUMNS, LABEL_COLUMN)

    holdback_ds = TensorDataset(
        torch.tensor(x_holdback, dtype=torch.float32),
        torch.tensor(y_holdback, dtype=torch.float32),
    )
    holdback_loader = DataLoader(holdback_ds, batch_size=args.batch_size, shuffle=False)

    model.load_state_dict(best_state["model_state_dict"])
    holdback_metrics = evaluate_model(model, holdback_loader, criterion, device)

    metrics_path = args.output_dir / "holdback_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in holdback_metrics.items()}, f, indent=2)
    print(f"Saved holdback metrics to: {metrics_path}")
    print("Holdback metrics:", {k: round(float(v), 5) for k, v in holdback_metrics.items()})


if __name__ == "__main__":
    main()
