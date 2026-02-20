import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

import evaluate
from datasets import Dataset, DatasetDict
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.task1.explorer import print_label_by_language
from src.task1.preprocessor import preprocess, SMM4HConfig
from src.utils import get_project_root, load_config

console = Console()
logger = logging.getLogger("task1_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ======= IMPORTANT: same threshold you used in training =======
THRESHOLD_DEFAULT = 0.30

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")


def load_data(cfg: dict):
    ds_cfg = cfg["dataset"]
    project_root = get_project_root()

    files = {
        "val": ds_cfg.get("val_filename", "val.csv"),
        "test": ds_cfg.get("test_filename", "test.csv"),
    }

    datasets = {}
    prep_config = SMM4HConfig(create_new_column=False, drop_empty_rows=True, deduplicate=True)

    for split, filename in files.items():
        path = project_root / ds_cfg["data_dir"] / filename
        if not path.exists():
            console.print(f"[red]Missing {split} at {path}[/red]")
            continue

        df = pd.read_csv(path)

        text_col = ds_cfg.get("text_column", "text")
        label_col = ds_cfg.get("label_column", "label")

        console.print(f"[bold cyan]Preprocessing {split} set...[/bold cyan]")
        df, _ = preprocess(
            df,
            text_column=text_col,
            label_column=label_col if label_col in df.columns else None,
            config=prep_config,
        )

        # language filter
        if "eval_lang" in ds_cfg and ds_cfg["eval_lang"] is not None:
            selected_langs = ds_cfg["eval_lang"]
            console.print(f"[yellow]Filtering {split} to languages: {selected_langs}[/yellow]")
            if "language" in df.columns:
                df = df[df["language"].isin(selected_langs)]
            else:
                console.print("[yellow]Warning: no 'language' column found, cannot filter.[/yellow]")

        rename_map = {}
        if text_col in df.columns:
            rename_map[text_col] = "text"
        if label_col in df.columns:
            rename_map[label_col] = "labels"
        df = df.rename(columns=rename_map)

        datasets[split] = Dataset.from_pandas(df)
        console.print(f"Loaded {split}: {len(df)} rows from {filename}")

        if "labels" in df.columns and "language" in df.columns:
            print_label_by_language(df=df, dataset_name=split, label_col="labels")

    return DatasetDict(datasets)


def tokenize_function(batch, tokenizer, max_length: int, truncation: bool):
    return tokenizer(batch["text"], truncation=truncation, max_length=max_length, padding=False)


def logits_to_pos_scores(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs[:, 1]
    logits_1d = logits.reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits_1d))


def print_threshold_sanity(scores, labels, threshold, title="Sanity"):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    pred = scores >= threshold
    pred_rate = float(pred.mean())
    true_rate = float(labels.mean())

    console.print(
        Panel(
            "\n".join(
                [
                    f"Threshold: {threshold:.3f}",
                    f"Scores: min={scores.min():.4f} | mean={scores.mean():.4f} | max={scores.max():.4f}",
                    f"Pred positives: {int(pred.sum())}/{len(labels)} ({pred_rate*100:.2f}%)",
                    f"True positives: {int(labels.sum())}/{len(labels)} ({true_rate*100:.2f}%)",
                ]
            ),
            title=title,
        )
    )


def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["NoADE", "ADE"],
        yticklabels=["NoADE", "ADE"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def compute_metrics_from_logits(logits, labels, threshold: float):
    labels = np.asarray(labels).astype(int)
    scores = logits_to_pos_scores(logits)
    preds = (scores >= threshold).astype(int)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)

    try:
        pr_auc_val = average_precision_score(labels, scores)
    except ValueError:
        pr_auc_val = 0.0

    try:
        roc_auc_val = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc_val = 0.0

    return {
        "accuracy": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "pr_auc": float(pr_auc_val),
        "roc_auc": float(roc_auc_val),
    }, scores, preds



def evaluate_split(trainer, ds, out_dir: Path, split_name: str, threshold: float):
    output = trainer.predict(ds)
    metrics, scores, preds = compute_metrics_from_logits(output.predictions, output.label_ids, threshold)

    print_threshold_sanity(scores, output.label_ids, threshold, title=f"{split_name} sanity")
    save_confusion_matrix(output.label_ids, preds, out_dir / f"{split_name}_confusion_matrix.png",
                          f"Confusion Matrix - {split_name}")

    # save raw predictions
    df = pd.DataFrame({"label": output.label_ids.astype(int), "pos_score": scores.astype(float), "pred": preds.astype(int)})
    df.to_csv(out_dir / f"{split_name}_predictions.csv", index=False)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1-config.json")
    ap.add_argument("--checkpoint", required=True, help="Path to final/ or checkpoint-XXXX folder")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT)
    ap.add_argument("--per_language", action="store_true")
    ap.add_argument("--save_dir", default=None, help="Default: <checkpoint>/eval")
    args = ap.parse_args()

    project_root = get_project_root()
    cfg = load_config(project_root / args.config)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    out_dir = Path(args.save_dir) if args.save_dir else checkpoint / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # data
    ds = load_data(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tok_cfg = cfg.get("tokenization", {})
    max_len = int(tok_cfg.get("max_length", 256))
    truncation = bool(tok_cfg.get("truncation", True))

    tokenized = ds.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_len, "truncation": truncation},
    )

    # model
    model_config = AutoConfig.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config=model_config)

    # trainer (eval only)
    ta = cfg.get("training_args", {})
    eval_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_eval_batch_size=int(ta.get("per_device_eval_batch_size", 32)),
        fp16=bool(ta.get("fp16", False)),
        bf16=bool(ta.get("bf16", False)),
        report_to=[],
        dataloader_drop_last=False,
        use_cpu=bool(ta.get("use_cpu", False)),
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    # evaluate
    report = {
        "checkpoint": str(checkpoint),
        "threshold": float(args.threshold),
        "val": {},
        "test": {},
    }

    table = Table(title=f"Re-evaluation @ threshold={args.threshold:.2f}")
    table.add_column("Split")
    table.add_column("F1", style="green")
    table.add_column("Precision")
    table.add_column("Recall")
    table.add_column("Accuracy")
    table.add_column("PR-AUC")
    table.add_column("ROC-AUC")

    val_m = evaluate_split(trainer, tokenized["val"], out_dir, "val", args.threshold)
    test_m = evaluate_split(trainer, tokenized["test"], out_dir, "test", args.threshold)

    report["val"] = val_m
    report["test"] = test_m

    for name, m in [("val", val_m), ("test", test_m)]:
        table.add_row(
            name,
            f"{m['f1']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['accuracy']:.4f}",
            f"{m['pr_auc']:.4f}",
            f"{m['roc_auc']:.4f}",
        )

    console.print(table)

    # per-language on test
    if args.per_language and "language" in tokenized["test"].column_names:
        lang_table = Table(title="Test metrics by language")
        lang_table.add_column("Language")
        lang_table.add_column("F1", style="green")
        lang_table.add_column("Precision")
        lang_table.add_column("Recall")
        lang_table.add_column("Accuracy")
        lang_table.add_column("PR-AUC")
        lang_table.add_column("ROC-AUC")

        report["test_by_language"] = {}
        langs = sorted(list(set(tokenized["test"]["language"])))
        for lang in langs:
            subset = tokenized["test"].filter(lambda x: x["language"] == lang)
            if len(subset) == 0:
                continue
            m = evaluate_split(trainer, subset, out_dir, f"test_{lang}", args.threshold)
            report["test_by_language"][lang] = m
            lang_table.add_row(
                lang,
                f"{m['f1']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['accuracy']:.4f}",
                f"{m['pr_auc']:.4f}",
                f"{m['roc_auc']:.4f}",
            )
        console.print(lang_table)

    # save
    with open(out_dir / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)

    # flat csv
    pd.DataFrame(
        [{"split": "val", **val_m}, {"split": "test", **test_m}]
    ).to_csv(out_dir / "evaluation_summary.csv", index=False)

    console.print(f"[bold green]Saved to {out_dir}[/bold green]")


if __name__ == "__main__":
    main()