from pathlib import Path
import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer
)
from datasets import Dataset

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
console = Console()

from src.utils import build_label_maps, load_config, print_config_summary
from src.task2.helper import print_dataset_examples, load_and_split_iris, get_dataset

import warnings
warnings.filterwarnings("ignore")

console = Console()
metric_acc = evaluate.load("accuracy")
metric_f1 = evaluate.load("f1")


def tokenize_function(batch, tokenizer, max_length: int, truncation: bool):
    return tokenizer(batch["text"], truncation=truncation, max_length=max_length)

def recover_text_from_tokens(tokenizer, ds, input_ids_col="input_ids"):
    return [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for ids in ds[input_ids_col]
    ]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
    f1 = metric_f1.compute(predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "macro_f1": f1}


def predict_decode(trainer, tokenizer, ds, label_col: str = None, add_probs: bool = False):
    if label_col is None:
        if "labels" in ds.column_names:
            label_col = "labels"
        elif "label" in ds.column_names:
            label_col = "label"
        else:
            raise KeyError("No label column found. Expected 'labels' or 'label'.")

    pred_out = trainer.predict(ds)
    logits = pred_out.predictions
    pred_ids = np.argmax(logits, axis=-1)

    true_ids = ds[label_col]
    id2label = getattr(trainer.model.config, "id2label", None)

    if id2label:
        true_labels = [id2label[int(i)] for i in true_ids]
        pred_labels = [id2label[int(i)] for i in pred_ids]
    else:
        true_labels = [int(i) for i in true_ids]
        pred_labels = [int(i) for i in pred_ids]

    texts = recover_text_from_tokens(tokenizer, ds)

    out = {"text": texts, "true_label": true_labels, "pred_label": pred_labels}

    if add_probs:
        x = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(x) / np.exp(x).sum(axis=-1, keepdims=True)
        out["probs"] = probs.tolist()

    return Dataset.from_dict(out)

def train_evaluate(dataset, cfg: dict, representation_type: str, id2label: dict, label2id: dict):
    model_name = cfg["model_name"]
    num_labels = int(cfg["num_labels"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    tok_cfg = cfg.get("tokenization", {})
    requested_max_len = int(tok_cfg.get("max_length", 512))
    truncation = bool(tok_cfg.get("truncation", True))
    model_max = getattr(tokenizer, "model_max_length", requested_max_len) or requested_max_len
    max_len = min(requested_max_len, int(model_max))

    console.print(Panel.fit(f"Tokenization: max_length={max_len}, truncation={truncation}", style="bold"))

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    if tokenizer.pad_token_id is not None and model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    console.rule("Tokenizing dataset", style="bold")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_len, "truncation": truncation},
        desc="Tokenizing",
    )

    keep = {"label", "input_ids", "attention_mask", "token_type_ids"}
    cols_to_remove = [c for c in tokenized["train"].column_names if c not in keep]
    if cols_to_remove:
        tokenized = tokenized.remove_columns(cols_to_remove)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    cfg["training_args"]["output_dir"] = str(Path("output/task2") / f"transformer-rep-{representation_type}")
    training_args = TrainingArguments(**cfg["training_args"])
    run_output_dir = cfg["training_args"]["output_dir"]

    callbacks = []
    es_cfg = cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(es_cfg.get("patience", 5))))
        console.print(f"[green]Early stopping enabled[/green] (patience={int(es_cfg.get('patience', 5))})")

    console.rule(f"Starting training (rep={representation_type})", style="bold green")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    trainer.train()
    console.rule("\nEvaluating", style="bold cyan")
    train_metrics = trainer.evaluate(tokenized["train"])
    val_metrics = trainer.evaluate(tokenized["val"])
    test_metrics = trainer.evaluate(tokenized["test"])

    test_with_preds = predict_decode(trainer, tokenizer, tokenized["test"], add_probs=False)

    t = Table(title=f"Metrics (rep={representation_type})", show_lines=True)
    t.add_column("Split", style="bold")
    t.add_column("accuracy")
    t.add_column("macro_f1")
    t.add_row("train", f"{train_metrics.get('eval_accuracy', float('nan')):.4f}", f"{train_metrics.get('eval_macro_f1', float('nan')):.4f}")
    t.add_row("val", f"{val_metrics.get('eval_accuracy', float('nan')):.4f}", f"{val_metrics.get('eval_macro_f1', float('nan')):.4f}")
    t.add_row("test", f"{test_metrics.get('eval_accuracy', float('nan')):.4f}", f"{test_metrics.get('eval_macro_f1', float('nan')):.4f}")
    console.print(t)

    return train_metrics, val_metrics, test_metrics, trainer, test_with_preds


def train_evaluate_experiments(dsA, dsB, id2label, label2id, cfg):

    print_examples = bool(cfg.get("print_examples", False))
    num_examples = int(cfg.get("num_examples", 1))

    if print_examples:
        print_dataset_examples(dsA, n=num_examples)
        print_dataset_examples(dsB, n=num_examples)

    merged_results = []
    test_preds ={}

    ROOT_DIR = Path(__file__).resolve().parents[2]
    out_dir = ROOT_DIR / "output/task2"

    for rep, ds in [("A", dsA), ("B", dsB)]:
        train_metrics, val_metrics, test_metrics, trainer, test_with_preds = train_evaluate(ds, cfg, rep, id2label, label2id)
        test_preds[rep] = test_with_preds
        out_path = f"{out_dir}/transformer_test_preds_rep{rep}.json"
        test_with_preds.to_json(out_path)
        merged_results.append(
            {
                "representation": rep,
                "sample": ds["test"][0],
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "best_global_step": trainer.state.best_global_step,
                "train": {"accuracy": train_metrics.get("eval_accuracy"), "macro_f1": train_metrics.get("eval_macro_f1")},
                "val": {"accuracy": val_metrics.get("eval_accuracy"), "macro_f1": val_metrics.get("eval_macro_f1")},
                "test": {"accuracy": test_metrics.get("eval_accuracy"), "macro_f1": test_metrics.get("eval_macro_f1")},
                "test_preds_file": out_path,
            }
        )
    results = {"transformer": {"results": merged_results}}

    return results, test_preds


if __name__ == "__main__":
    iris, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_iris(random_state=42, test_size=0.1, val_size=0.111)
    dsA = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
    dsB = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test)
    cfg = load_config(".../task2-config.json")
    num_labels = int(cfg["num_labels"])
    print_config_summary(cfg,num_labels)
    id2label, label2id = build_label_maps(cfg=cfg, num_labels=num_labels)
    results = train_evaluate_experiments(dsA, dsB, id2label, label2id, cfg)
