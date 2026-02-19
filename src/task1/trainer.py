import json
import sys
import time
from datetime import datetime
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback, TrainerCallback
 )
import evaluate
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Rich for pretty output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.task1.explorer import print_label_by_language
from src.task1.preprocessor import preprocess, SMM4HConfig
from src.utils import get_project_root, load_config



# Setup
console = Console()
logger = logging.getLogger("task1_trainer")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

THRESHOLD = 0.15 # 25


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

class ClassWeightedTrainer(Trainer):
    """
    Custom Trainer that overrides compute_loss to handle class imbalance.
    """

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure weights are on the correct device
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
            self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss



class TimingCallback(TrainerCallback):
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.train_start = None
        self.epoch_start = None
        self.epoch_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.perf_counter()
        self.epoch_times = []
        self.logger.info("Training started.")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()
        self.logger.info(f"Epoch {state.epoch + 1} started.")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start is None:
            return
        dur = time.perf_counter() - self.epoch_start
        self.epoch_times.append({"epoch": state.epoch, "seconds": dur})
        self.logger.info(f"Epoch {state.epoch} finished in {dur:.2f}s.")

    def on_train_end(self, args, state, control, **kwargs):
        if self.train_start is None:
            return
        total = time.perf_counter() - self.train_start
        self.logger.info(f"Training finished in {total:.2f}s.")


def load_data(cfg: dict):
    """
    Loads data from CSVs specified in config.
    Filters by language if 'lang_filter' is provided.
    """
    ds_cfg = cfg["dataset"]
    project_root = get_project_root()
    files = {
        "train": ds_cfg.get("train_filename", "train.csv"),
        "val": ds_cfg.get("val_filename", "val.csv"),
        "test": ds_cfg.get("test_filename", "test.csv")
    }
    datasets = {}
    prep_config = SMM4HConfig(
        create_new_column=False,
        drop_empty_rows=True,
        deduplicate=True
    )
    for split, filename in files.items():
        if "data_dir" in ds_cfg:
            path = project_root / ds_cfg["data_dir"] / filename
        else:
            path = project_root / filename
        if path.exists():
            df = pd.read_csv(path)

            # Ensure text is string and handle NaNs
            text_col = ds_cfg.get("text_column", "text")
            label_col = ds_cfg.get("label_column", "label")
            if text_col in df.columns:
                console.print(f"[bold cyan]Preprocessing {split} set...[/bold cyan]")

                df, _ = preprocess(
                    df,
                    text_column=text_col,
                    label_column=label_col if label_col in df.columns else None,
                    config=prep_config
                )
            if split == "train" and ds_cfg.get("multi_trans", False):
                if "text_source" in df.columns:
                    console.print(
                        f"[bold cyan]Preprocessing 'text_source' (English Source) in {split}...[/bold cyan]")


                    df, _ = preprocess(
                        df,
                        text_column="text_source",
                        config=SMM4HConfig(
                            create_new_column=False,
                            drop_empty_rows=True,
                            deduplicate=False
                        )
                    )
                else:
                    console.print(f"[red]Warning: 'source_text' missing for multi_trans![/red]")

            # TRAIN SPLIT LOGIC
            if split == "train":
                if ds_cfg.get("train_lang") is not None:
                    selected_langs = ds_cfg["train_lang"]
                    console.print(f"[yellow]Filtering {split} set to languages: {selected_langs}[/yellow]")
                    df = df[df['language'].isin(selected_langs)]

                    if "sample_frac_zero" in ds_cfg and ds_cfg.get("train_lang") is not None:
                        frac = float(ds_cfg["sample_frac_zero"])
                        if 0.0 < frac < 1.0:
                            current_label_col = ds_cfg.get("label_column", "label")
                            df_pos = df[df[current_label_col] == 1]
                            df_neg = df[df[current_label_col] == 0]
                            console.print(f"[yellow]Downsampling ONLY Label 0 to {frac * 100}% of its size...[/yellow]")
                            df_neg_sampled = df_neg.sample(frac=frac, random_state=42)
                            df = pd.concat([df_pos, df_neg_sampled]).sample(frac=1, random_state=42).reset_index(
                                drop=True)
                            console.print(
                                f"[bold green]New Train Distribution -> Label 0: {len(df_neg_sampled)} | Label 1: {len(df_pos)}[/bold green]")

                else:
                    if ds_cfg.get("multi_trans", False):
                        console.print(f"[blue]Combining Translated (De) and Source (En) data...[/blue]")

                        df_trans = df[[text_col, ds_cfg["label_column"]]].copy()
                        df_trans["language"] = "de_trans"  # for tracking

                        df_source = df[["text_source", ds_cfg["label_column"]]].copy()
                        df_source = df_source.rename(columns={"text_source": text_col})
                        df_source["language"] = "en_source"  # for tracking

                        df = pd.concat([df_trans, df_source], ignore_index=True)

                    else:
                        console.print(f"[blue]Using Translated Data (German Only)[/blue]")
                        df["language"] = "de_trans"


            if split != "train" and "eval_lang" in ds_cfg:
                selected_langs = ds_cfg["eval_lang"]
                console.print(f"[yellow]Filtering {split} set to languages: {selected_langs}[/yellow]")
                df = df[df['language'].isin(selected_langs)]

            # Rename columns to standard HF names
            rename_map = {}
            if text_col in df.columns:
                rename_map[text_col] = "text"
            if ds_cfg["label_column"] in df.columns:
                rename_map[ds_cfg["label_column"]] = "labels"

            df = df.rename(columns=rename_map)

            datasets[split] = Dataset.from_pandas(df)
            console.print(f"Loaded {split}: {len(df)} rows from {filename}")
            print_label_by_language(df=df, dataset_name=split, label_col="labels")
        else:
            console.print(f"[red]Warning: {split} file not found at {path}[/red]")

    return DatasetDict(datasets)


def tokenize_function(batch, tokenizer, max_length: int, truncation: bool):
    return tokenizer(batch["text"], truncation=truncation, max_length=max_length, padding=False # Padding handled by DataCollator
     )

def compute_metrics_classic(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def compute_metrics_(eval_pred):
    logits, labels = eval_pred

    if isinstance(logits, tuple):
        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1)["precision"]
    recall = recall_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1)["recall"]
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1)["f1"]

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    labels = np.asarray(labels)

    # --- SET YOUR CUSTOM THRESHOLD HERE ---
    #THRESHOLD = 0.15  # Lowered from 0.5 due to 7% class imbalance

    # Case A: model outputs 2 logits per sample: shape (N, 2)
    if logits.ndim == 2 and logits.shape[1] == 2:
        # softmax -> prob of class 1
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        pos_scores = probs[:, 1]

        # FIX: Do not use argmax. Use the threshold against pos_scores.
        preds = (pos_scores >= THRESHOLD).astype(int)

    # Case B: model outputs 1 logit per sample: shape (N,) or (N,1)
    else:
        logits_1d = logits.reshape(-1)
        # sigmoid -> prob of class 1
        pos_scores = 1.0 / (1.0 + np.exp(-logits_1d))

        # FIX: Use custom threshold instead of 0.5
        preds = (pos_scores >= THRESHOLD).astype(int)

    try:
        pr_auc_val = average_precision_score(labels, pos_scores)
    except ValueError:
        pr_auc_val = 0.0

    try:
        roc_auc_val = roc_auc_score(labels, pos_scores)
    except ValueError:
        roc_auc_val = 0.0

    out = {
        "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1,
                                              zero_division=0)["precision"],
        "recall":
            recall_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1, zero_division=0)[
                "recall"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="binary", pos_label=1)["f1"],
        "pr_auc": pr_auc_val,
        "roc_auc": roc_auc_val  # optional
    }
    return out

def save_confusion_matrix(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=["NoADE", "ADE"], yticklabels=["NoADE", "ADE"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def get_metric(metrics_dict, name):
    # Try "test_f1", "eval_f1", then just "f1"
    for prefix in ["test_", "eval_", ""]:
        key = f"{prefix}{name}"
        if key in metrics_dict:
            return metrics_dict[key]
    return float("nan")

def logits_to_pos_scores(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 2 and logits.shape[1] == 2:
        exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        return probs[:, 1]
    logits_1d = logits.reshape(-1)
    return 1.0 / (1.0 + np.exp(-logits_1d))

def find_best_threshold(y_true, pos_scores):
    best = {"t": 0.5, "f1": -1, "p": 0, "r": 0}
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (pos_scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"t": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best


def print_threshold_sanity(console, scores, labels, threshold, title="Score / Threshold Sanity Check"):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    pred_mask = scores >= threshold
    pred_rate = float(pred_mask.mean())
    true_rate = float(labels.mean())

    pred_pos = int(pred_mask.sum())
    true_pos = int(labels.sum())
    n = int(len(labels))

    over_factor = pred_rate / (true_rate + 1e-12)

    console.print(Panel(
        "\n".join([
            f"Threshold: {threshold:.3f}",
            f"Scores: min={scores.min():.4f} | mean={scores.mean():.4f} | max={scores.max():.4f}",
            f"Predicted positives: {pred_pos}/{n} ({pred_rate*100:.2f}%)",
            f"True positives:      {true_pos}/{n} ({true_rate*100:.2f}%)",
            f"Over prediction:     {over_factor:.2f}x",
        ]),
        title=title
    ))

def run_training(config_path: str):
    # Load Config
    project_root = get_project_root()
    cfg = load_config(project_root / config_path)

    model_name = cfg["model_name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = project_root / cfg["output_dir"] / model_name / timestamp
    console.rule(f"Running Experiment: {model_name}", style="bold green")

    # Load Data
    dataset = load_data(cfg)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tok_cfg = cfg.get("tokenization", {})
    requested_max_len = int(tok_cfg.get("max_length", 512))
    truncation = bool(tok_cfg.get("truncation", True))
    model_max = getattr(tokenizer, "model_max_length", requested_max_len) or requested_max_len
    max_len = min(requested_max_len, int(model_max))

    tokenized_ds = dataset.map(tokenize_function, batched=True , fn_kwargs={"tokenizer": tokenizer, "max_length": max_len, "truncation": truncation})

    # Class Weights (for Imbalance)
    # Class Weights (Softer)
    class_weights = None
    if cfg["imbalance_handling"]["use_class_weights"]:
        # Get labels
        train_labels = tokenized_ds["train"]["label"] if "label" in tokenized_ds["train"].column_names else \
        tokenized_ds["train"]["labels"]
        labels = np.array(train_labels)

        # Compute standard balanced weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(labels),
            y=labels
        )

        # Take the square root to make the ratio less extreme
        # Example: Instead of 1:12, it becomes 1:3.5
        #class_weights = np.sqrt(class_weights)

        # Alternative: Hardcap the positive weight (e.g., max 5.0)
        # class_weights[1] = min(class_weights[1], 5.0)

        console.print(
            Panel(f"Weights: {class_weights}", title="Imbalance Handling"))
    # Model
    id2label = cfg["id2label"]
    label2id = cfg["label2id"]

    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=cfg["num_labels"],
        id2label=id2label,
        label2id=label2id,
        #hidden_dropout_prob=0.3,  # from the paper!
        #attention_probs_dropout_prob=0.3  # from the paper!,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)

    # Trainer Setup
    train_args = TrainingArguments(
        output_dir=str(output_dir),
        **cfg["training_args"]
    )

    collator = DataCollatorWithPadding(tokenizer)

    callbacks = []
    timing_cb = TimingCallback(logger=logger)
    callbacks.append(timing_cb)
    es_cfg = cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(es_cfg.get("patience", 5))))
        console.print(f"[green]Early stopping enabled[/green] (patience={int(es_cfg.get('patience', 5))})")

    trainer = ClassWeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=train_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )

    # Train
    console.print("[bold]Starting Training...[/bold]")
    trainer.train()

    total_seconds = sum(x["seconds"] for x in timing_cb.epoch_times)
    console.print(Panel(
        "\n".join([f"Epoch {x['epoch']}: {x['seconds']:.2f}s" for x in timing_cb.epoch_times]) +
        f"\n\nTotal Time: {total_seconds:.2f}s",
        title="Training Time"
    ))
    # Save Final Model
    trainer.save_model(str(output_dir / "final"))
    console.print(f"[bold green]Model saved to {output_dir}/final[/bold green]")

    # --------------------------------------------
    # Evaluate & Save
    console.rule("Evaluation & Analysis", style="bold cyan")
    final_report = {}
    final_report["timing"] = {
        "epoch_times": timing_cb.epoch_times,
        "total_time": total_seconds
    }
    t_global = Table(title=f"Overall Performance: {model_name} Model")
    t_global.add_column("Split")
    t_global.add_column("F1-Score", style="green")
    t_global.add_column("Precision")
    t_global.add_column("Recall")
    t_global.add_column("Accuracy")
    t_global.add_column("PR-AUC")
    t_global.add_column("ROC-AUC")


    for split_name, ds in [("Val", tokenized_ds["val"]), ("Test", tokenized_ds["test"])]:
        console.print(f"[bold]Evaluating {split_name} Split: [/bold]")
        output = trainer.predict(ds)
        scores = logits_to_pos_scores(output.predictions)
        labels = output.label_ids

        #best = find_best_threshold(labels, scores)
        #THRESHOLD = best["t"]

        preds = (scores >= THRESHOLD).astype(int)
        print_threshold_sanity(scores, labels, THRESHOLD, title=f"{split_name} Sanity Check")

        # Save Confusion Matrix
        save_confusion_matrix(
            labels, preds,
            output_dir / f"{split_name.lower()}_confusion_matrix.png",
            f"Confusion Matrix - {split_name}"
        )

        m = output.metrics
        t_global.add_row(
            split_name,
            f"{get_metric(m, 'f1'):.4f}",
            f"{get_metric(m, 'precision'):.4f}",
            f"{get_metric(m, 'recall'):.4f}",
            f"{get_metric(m, 'accuracy'):.4f}",
            f"{get_metric(m, 'pr_auc'):.4f}",
            f"{get_metric(m, 'roc_auc'):.4f}",
        )
        final_report[f"global_{split_name.lower()}"] = m
    console.print(t_global)
    # --------------------------------------------
    console.print("\n")

    t_lang = Table(title="Test Set Performance by Language")
    t_lang.add_column("Language")
    t_lang.add_column("F1-Score", style="green")
    t_lang.add_column("Precision")
    t_lang.add_column("Recall")
    t_lang.add_column("Accuracy")
    t_lang.add_column("PR-AUC")
    t_lang.add_column("ROC-AUC")

    unique_langs = sorted(list(set(tokenized_ds["test"]["language"])))

    for lang in unique_langs:
        console.print(f"[bold]Evaluating Language: {lang}[/bold]")
        lang_subset = tokenized_ds["test"].filter(lambda x: x["language"] == lang)
        if len(lang_subset) == 0:
            console.print(f"[yellow]Skipping {lang} (No samples)[/yellow]")
            continue
        lang_output = trainer.predict(lang_subset)
        lang_scores = logits_to_pos_scores(lang_output.predictions)
        lang_labels = lang_output.label_ids

        lang_preds = (lang_scores >= THRESHOLD).astype(int)

        save_confusion_matrix(
            lang_labels, lang_preds,
            output_dir / f"test_{lang.lower()}_confusion_matrix.png",
            f"Confusion Matrix - Test ({lang})"
        )

        m = lang_output.metrics
        t_lang.add_row(
            lang,
            f"{get_metric(m, 'f1'):.4f}",
            f"{get_metric(m, 'precision'):.4f}",
            f"{get_metric(m, 'recall'):.4f}",
            f"{get_metric(m, 'accuracy'):.4f}",
            f"{get_metric(m, 'pr_auc'):.4f}",
            f"{get_metric(m, 'roc_auc'):.4f}",
        )
        final_report[f"test_lang_{lang}"] = m

    console.print(t_lang)
    with open(output_dir / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=4)

    console.print(f"[bold green]Evaluation complete. Results saved to {output_dir}[/bold green]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "configs/task1-config.json"

    run_training(config_file)