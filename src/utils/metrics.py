from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


SUMMARY_COLUMNS = [
    "scenario",
    "dataset",
    "method",
    "backbone",
    "accuracy",
    "top1",
    "k",
    "feature_dim",
]


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_path)


def save_classification_table(
    paths: Iterable[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, true_idx, pred_idx in zip(paths, y_true, y_pred):
        rows.append(
            {
                "path": path,
                "label": int(true_idx),
                "label_name": class_names[int(true_idx)],
                "prediction": int(pred_idx),
                "prediction_name": class_names[int(pred_idx)],
                "correct": int(true_idx == pred_idx),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


def append_summary(row: dict, summary_csv: str | Path = "results/summary.csv") -> None:
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})
