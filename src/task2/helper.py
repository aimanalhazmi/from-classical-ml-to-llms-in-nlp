import hashlib
from pathlib import Path
import json
import numpy as np
import re
from datetime import datetime, timezone
import ast
from typing import Any

from jupyterlab.labapp import clean_flags
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from datasets import Dataset, DatasetDict

from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel

console = Console()
def load_and_split_iris(random_state=42, test_size=0.1, val_size=0.111):
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random_state
    )
    return iris, X_train, y_train, X_val, y_val, X_test, y_test


def textify_iris(features, representation_type="A"):
    text_data = []
    feat_names = ["sepal length", "sepal width", "petal length", "petal width"]
    for row in features.itertuples(index=False):
        if representation_type == "A":
            text = (
                f"the flower has a {feat_names[0]} of {row[0]}cm, a {feat_names[1]} of {row[1]}cm, "
                f"a {feat_names[2]} of {row[2]}cm, and a {feat_names[3]} of {row[3]}cm."
            )
        else:
            text = f"sl:{row[0]} sw:{row[1]} pl:{row[2]} pw:{row[3]}"
        text_data.append(text)
    return text_data

def get_dataset(X_train, y_train, X_val, y_val, X_test, y_test, representation_type="A", print_examples=False):
    train_text = textify_iris(features=X_train, representation_type=representation_type)
    val_text = textify_iris(features=X_val, representation_type=representation_type)
    test_text = textify_iris(features=X_test, representation_type=representation_type)

    if len(train_text) != len(y_train):
        raise ValueError("Train text/label length mismatch")
    if len(val_text) != len(y_val):
        raise ValueError("Val text/label length mismatch")
    if len(test_text) != len(y_test):
        raise ValueError("Test text/label length mismatch")

    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": train_text, "label": y_train}),
            "val": Dataset.from_dict({"text": val_text, "label": y_val}),
            "test": Dataset.from_dict({"text": test_text, "label": y_test}),
        }
    )

    if print_examples:
        console.print(f"Representation {representation_type} examples:")
        for split in ["train", "val", "test"]:
            ex = dataset[split][0]
            console.print(f"\n[bold]{split}[/bold] label={ex['label']}\n{ex['text']}")

    return dataset

def save_results(out_path: str | Path, results: dict | list, cfg: dict | None = None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    is_transformer = isinstance(results, dict) and "transformer" in results
    cfg_payload = cfg if (cfg is not None and is_transformer) else {}

    payload = json.dumps({"cfg": cfg_payload, "results": results}, sort_keys=True, default=str).encode("utf-8")
    run_id = hashlib.sha1(payload).hexdigest()[:10]

    merged = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **({"config": cfg_payload} if cfg_payload else {}),
        "experiments": results,
    }

    out_path.write_text(json.dumps(merged, indent=2, sort_keys=True, default=str), encoding="utf-8")
    console.print(f"[green]Saved results:[/green] {out_path} (run_id={run_id})")
    return merged


def validate_llmOutput_return_json(output: Any):
    if isinstance(output, (dict, list)):
        return output

    if not isinstance(output, str):
        raise ValueError(f"Expected str/dict/list, got {type(output)}")

    s = output.strip()

    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        s = s[start:end+1].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(f"Invalid JSON Format: {output!r}")


def print_dataset_examples(dataset, n: int = 1):
    for split in ["train", "val", "test"]:
        if split not in dataset:
            continue
        console.print(Panel.fit(f"Examples: {split}", style="bold"))
        for i in range(min(n, len(dataset[split]))):
            ex = dataset[split][i]
            console.print(f"[bold]{split}[{i}] label={ex.get('label')}[/bold]\n{ex.get('text')}\n")


from typing import Dict, Any
from rich.table import Table

def print_classification_report_split_table(split_name: str, summary: Dict[str, Any]) -> None:
    """Pretty-print metrics for a split (Rich table)."""
    meta = summary.get("metadata", {})
    report = summary.get("classification_report", {})

    table = Table(title=f"Performance Metrics — {split_name}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", justify="left", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Requested Samples", str(meta.get("requested_num_samples", 0)))
    table.add_row("Evaluated Samples", str(meta.get("evaluated_num_samples", 0)))
    table.add_row("Correct Predictions", f"[green]{meta.get('total_correct', 0)}[/green]")
    table.add_row("Accuracy", f"{report.get('accuracy', 0.0):.2%}")
    table.add_row("Macro F1-Score", f"{report.get('macro avg', {}).get('f1-score', 0.0):.4f}")
    table.add_row("Weighted F1-Score", f"{report.get('weighted avg', {}).get('f1-score', 0.0):.4f}")

    table.add_section()
    table.add_row("Total Inference Time", f"{meta.get('total_inference_time', 0.0):.2f}s")
    table.add_row("Avg Time per Evaluated Sample", f"{meta.get('avg_time_per_sample', 0.0):.4f}s")

    console.print(table)

def print_metrics_tables_by_split(clf_results: dict, tf_results: list[dict], llm_results: dict):
    dt = clf_results.get("decision_tree", {})
    for split in ["train", "val", "test"]:
        t = Table(title=f"Metrics ({split})", show_lines=True)
        t.add_column("Model", style="bold")
        t.add_column("accuracy")
        t.add_column("macro_f1")

        # DecisionTree row
        s = dt.get(split, {})
        t.add_row(
            "DecisionTree",
            f"{s.get('accuracy', float('nan')):.4f}",
            f"{s.get('macro_f1', float('nan')):.4f}",
        )

        # Transformer rows
        for r in tf_results:
            rep = r.get("representation", "?")
            m = r.get(split, {})

            if m:
                t.add_row(
                    f"Transformer({rep})",
                    f"{m.get('accuracy', float('nan')):.4f}",
                    f"{m.get('macro_f1', float('nan')):.4f}",
                )

        # LLM-Based Prediction and Hybrid Modeling
        # llm_summary = {"llm_predictions": {"llm_without_rules": llmTrainResultsNoRules, "llm_with_rules": llmTrainResultsWithRules}}
        for k , v in llm_results["llm_predictions"].items():
            if k.lower() == "llm_without_rules":
                display_name = ""
            else:
                display_name = "(With classical model information)"

            llm_test_results = v.get(split, {})
            if llm_test_results:
                t.add_row(
                    f"LLM {display_name}",
                    f"{llm_test_results.get('accuracy', float('nan')):.4f}",
                    f"{llm_test_results.get('macro_f1', float('nan')):.4f}",

                )
        console.print(t)

def extract_features_from_text(text_str):
    """
    Parses 'sl : 6. 6 sw : 2. 9...' into a numpy array [6.6, 2.9, ...]
    """
    clean_str = text_str.replace(" ", "")
    matches = re.findall(r"(\d+\.\d+)", clean_str)
    if len(matches) == 4:
        return np.array([float(x) for x in matches])
    else:
        return np.zeros((1, 4))

def check(true_val, pred): return "✅" if str(pred).strip().lower() == str(true_val).strip().lower() else "❌"

def comparative_analysis(clf, id2label, clf_results, tf_results, tf_test_preds, llm_results):
    console.rule("\nComparative Analysis", style="bold cyan")
    print_metrics_tables_by_split(clf_results=clf_results, tf_results=tf_results, llm_results=llm_results)
    tf_data = tf_test_preds.get("B")

    # LLM Data
    llm_no_rules_data = llm_results["llm_predictions"].get("llm_without_rules", {}).get("test", {}).get(
        "detailed_results", [])
    llm_with_rules_data = llm_results["llm_predictions"].get("llm_with_rules", {}).get("test", {}).get(
        "detailed_results", [])

    length = min(len(tf_data), len(llm_with_rules_data))

    clf_features = []
    for i in range(length):
        feat = extract_features_from_text(tf_data[i]["text"])
        clf_features.append(feat)
    clf_preds_indices = clf.predict(clf_features)

    for i in range(length):

        # Transformer
        tf_row = tf_data[i]
        tf_text = tf_row['text']
        tf_true = tf_row['true_label']
        tf_pred = tf_row['pred_label']

        # LLM (No Rules)
        llm_nr_row = llm_no_rules_data[i]
        llm_nr_text = llm_nr_row['text']
        llm_nr_true = llm_nr_row['true_label']
        llm_nr_pred = llm_nr_row['predicted_label']

        # LLM (With Rules)
        llm_wr_row = llm_with_rules_data[i]
        llm_wr_text = llm_wr_row['text']
        llm_wr_true = llm_wr_row['true_label']
        llm_wr_pred = llm_wr_row['predicted_label']

        clf_pred = id2label[clf_preds_indices[i]].lower()

        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("Model", style="cyan", width=20)
        table.add_column("True Label", style="green")
        table.add_column("Predicted Label", style="yellow")
        table.add_column("Match?", justify="center")



        table.add_row("Decision Tree", str(tf_true), str(clf_pred), check(tf_true, clf_pred))
        table.add_row("Transformer ", str(tf_true), str(tf_pred), check(tf_true, tf_pred))
        table.add_row("LLM (No Rules)", str(llm_nr_true), str(llm_nr_pred), check(llm_nr_true, llm_nr_pred))
        table.add_row("LLM (With Rules)", str(llm_wr_true), str(llm_wr_pred), check(llm_wr_true, llm_wr_pred))


        # Displaying both texts as requested to verify they match
        text_display = Text()
        text_display.append("Text Comparison:\n", style="bold underline")
        text_display.append(f"Decision Tree: {clf_features[i]}\n", style="dim")
        text_display.append(f"TF:  {tf_text}\n", style="dim")
        text_display.append(f"LLM: {llm_nr_text}\n", style="dim")


        # Group combines the text and the table vertically
        panel_content = Group(
            text_display,
            Text(" "),  # Spacer
            table
        )

        console.print(Panel(
            panel_content,
            title=f"Sample Index [{i}]",
            subtitle="Model Comparison",
            border_style="blue"
        ))
    results = [clf_results, tf_results, llm_results]
    save_results("output/task2/summary.json", results=results, cfg=None)