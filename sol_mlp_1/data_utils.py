from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LabelEncoderState:
    mapping: dict[str, int]
    inverse_mapping: dict[int, str]


def load_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        suffix = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise ValueError(f"Missing required columns: {preview}{suffix}")


def encode_binary_labels(series: pd.Series) -> tuple[np.ndarray, LabelEncoderState]:
    non_null = series.dropna()
    unique_vals = sorted(non_null.unique().tolist())
    if len(unique_vals) != 2:
        raise ValueError(
            "Label column must contain exactly two classes after dropping NaN values. "
            f"Found {len(unique_vals)} classes: {unique_vals}"
        )

    mapping = {str(unique_vals[0]): 0, str(unique_vals[1]): 1}
    encoded = series.astype(str).map(mapping).to_numpy(dtype=np.float32)

    if np.isnan(encoded).any():
        raise ValueError("Found labels that could not be encoded into 0/1.")

    inverse = {v: k for k, v in mapping.items()}
    return encoded, LabelEncoderState(mapping=mapping, inverse_mapping=inverse)


def get_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
) -> tuple[np.ndarray, np.ndarray, LabelEncoderState]:
    validate_columns(df, feature_columns + [label_column])

    features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    if features.isnull().any().any():
        bad_cols = features.columns[features.isnull().any()].tolist()
        raise ValueError(
            "Feature conversion produced NaN values. Check source data for missing/non-numeric entries in: "
            + ", ".join(bad_cols)
        )

    y, encoder_state = encode_binary_labels(df[label_column])
    x = features.to_numpy(dtype=np.float32)
    return x, y, encoder_state


# def split_train_val(
#     x: np.ndarray,
#     y: np.ndarray,
#     val_size: float,
#     random_seed: int,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     return train_test_split(
#         x,
#         y,
#         test_size=val_size,
#         random_state=random_seed,
#         stratify=y,
#     )
