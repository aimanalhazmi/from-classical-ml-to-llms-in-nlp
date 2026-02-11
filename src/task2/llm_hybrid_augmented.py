from pathlib import Path
from tqdm.auto import tqdm
import time
import random
from sklearn.metrics import classification_report, accuracy_score, f1_score

from collections import defaultdict
from typing import Optional, Dict, Any, List
from rich.console import Console
from rich.panel import Panel

from src.llm import Client as LLMClient
from src.utils import load_config, print_config_summary, build_label_maps
from src.task2.helper import validate_llmOutput_return_json, load_and_split_iris, get_dataset, save_results, print_classification_report_split_table

console = Console()

SYSTEM_MESSAGE_NO_RULES = """You are a professional botanical classification assistant. 
You will be provided with a set of Flower Measurements (SL: Sepal Length, SW: Sepal Width, PL: Petal Length, PW: Petal Width). 
Your task is to classify the flower into one of three species: setosa, versicolor, or virginica.
*Output Format:* You must respond strictly in JSON format with the following structure: 
{{ 'predicted_label': <string>, 'confidence': <float between 0 and 1> }}
"""

SYSTEM_MESSAGE_WITH_RULES = """You are a professional botanical classification assistant. 
You will be provided with Decision Rules and Feature importance (extracted from a Decision Tree model) and a set of Flower Measurements (SL: Sepal Length, SW: Sepal Width, PL: Petal Length, PW: Petal Width). 
Your task is to use the provided rules as a primary logic gate to classify the flower into one of three species: setosa, versicolor, or virginica.

*Output Format:* You must respond strictly in JSON format with the following structure: 
{{"predicted_label": "...", "confidence": 0.0}}
No preamble, no markdown formatting, no explanations.
"""
def formatUserMessage(
    sample: str,
    rules: Optional[str] = None,
    importance: Optional[str] = None,
    few_shot_examples: Optional[list[dict]] = None
) -> str:
    userMessage = "Analyze the measurements against the rules and examples provided and return only the classification JSON.\n"

    has_rules = rules is not None and rules.strip() != ""
    has_importance =  importance is not None and importance.strip() != ""

    if has_rules and has_importance:
        userMessage += \
            f"""### Decision Rules ###\n {rules} \n### Feature importances ###\n {importance}\n"""
    if few_shot_examples:
        userMessage += "FEW-SHOT EXAMPLES:\n"
        for ex in few_shot_examples:
            userMessage += f"- {ex['text']} => {ex['label_name']}\n"
        userMessage += "\n"

    userMessage += \
    f"""### MEASUREMENTS TO CLASSIFY ###
    {sample}\n### INSTRUCTION ### \n
*Output Format:* {{"predicted_label": "...", "confidence": 0.0}}
No preamble, no markdown formatting, no explanations.
"""
    return userMessage

def evaluate_sample_with_llm(
    llm, sample,
    rules: Optional[str] = None,
    importances: Optional[str] = None,
    few_shot_examples: Optional[list[dict]] = None
):
    has_rules = rules is not None and rules.strip() != ""
    has_importances = importances is not None and importances.strip() != ""

    systemMessage = SYSTEM_MESSAGE_WITH_RULES if (has_rules and has_importances) else SYSTEM_MESSAGE_NO_RULES

    userMessage = formatUserMessage(sample=sample, rules=rules, importance=importances, few_shot_examples=few_shot_examples)

    return llm.generate(system_message=systemMessage, user_message=userMessage)


def get_few_shot_samples(dataset, id2label, n_shots=3, label_col: str = "label",
    text_col: str = "text", seed: Optional[int] = None):
    """Selects n_shots examples for each class from the dataset."""
    rng = random.Random(seed)
    labels = dataset[label_col]
    texts = dataset[text_col]

    label_to_indices = defaultdict(list)
    for i, y in enumerate(labels):
        label_to_indices[y].append(i)

    label_ids = sorted(label_to_indices.keys())

    chosen = []
    few_shot = []

    for y in label_ids:
        inds = label_to_indices[y]
        k = min(n_shots, len(inds))
        if k == 0:
            continue

        picked = rng.sample(inds, k)
        chosen.extend(picked)

        label_name = id2label.get(y, str(y)).lower()
        for idx in picked:
            few_shot.append({"text": texts[idx], "label_name": label_name})

    chosen_set = set(chosen)
    remaining_indices = [i for i in range(len(dataset)) if i not in chosen_set]
    return few_shot, dataset.select(remaining_indices)


def evaluate_data_split_with_llm(
    *,
    llm,
    split_name: str,
    data,
    num_samples: int,
    id2label: Dict[int, str],
    few_shots: Optional[List[dict]] = None,
    rules: Optional[str] = None,
    importances: Optional[str] = None,
    show_table: bool = True,
) -> Dict[str, Any]:

    n = min(num_samples, len(data))
    if n == 0:
        summary = {
            "metadata": {
                "split": split_name,
                "requested_num_samples": num_samples,
                "evaluated_num_samples": 0,
                "total_correct": 0,
                "has_rules": bool(rules and rules.strip()),
                "has_importance": bool(importances and importances.strip()),
                "total_inference_time": 0.0,
                "avg_time_per_sample": 0.0,
            },
            "classification_report": {},
            "detailed_results": [],
        }
        if show_table:
            print_classification_report_split_table(split_name, summary)
        return summary

    if "orig_idx" not in data.column_names:
        data = data.add_column("orig_idx", (range(len(data))))

    samples_to_test = data.select(range(n))

    y_pred: List[str] = []
    y_true: List[str] = []
    total_correct = 0
    detailed_logs: List[dict] = []

    start_time = time.time()

    for sample in tqdm(samples_to_test, desc=f"LLM Inference ({split_name})"):
        orig_idx = sample["orig_idx"]
        true_label = id2label[sample["label"]].lower()
        feature_text = sample["text"]

        raw_output = evaluate_sample_with_llm(
            llm=llm,
            sample=feature_text,
            rules=rules,
            importances=importances,
            few_shot_examples=few_shots,
        )

        try:
            llm_json = validate_llmOutput_return_json(output=raw_output)
            predicted_label = str(llm_json.get("predicted_label", "unknown")).lower()
            confidence = float(llm_json.get("confidence", 0.0))
        except Exception as e:
            console.print(Panel(f"[{split_name}] Validation failed for ID {orig_idx}: {e}", border_style="bold yellow"))
            continue

        is_correct = (predicted_label == true_label)
        total_correct += int(is_correct)

        y_pred.append(predicted_label)
        y_true.append(true_label)

        if split_name.lower() == "test":
            status_color = "green" if is_correct else "red"
            panel_content = (
                f"[bold]Input Data:[/bold] {feature_text}\n" f"[bold]Ground Truth:[/bold] {true_label}\n" f"[{status_color}][bold]LLM Prediction:[/bold] {predicted_label} ({confidence:.2f})[/]")
            console.print(Panel(panel_content, title=f"Sample #{orig_idx}", border_style=status_color))

            detailed_logs.append({
                "orig_idx": orig_idx,
                "text": feature_text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": is_correct,
            })

    total_time = time.time() - start_time
    avg_time = total_time / max(1, len(y_true))

    labels = [id2label[i].lower() for i in sorted(id2label.keys())]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    summary = {
        "metadata": {
            "requested_num_samples": num_samples,
            "evaluated_num_samples": len(y_true),
            "total_correct": total_correct,
            "has_rules": bool(rules and rules.strip()),
            "has_importance": bool(importances and importances.strip()),
            "total_inference_time": total_time,
            "avg_time_per_sample": avg_time,
        },
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
        "detailed_results": detailed_logs,
    }

    if show_table:
        print_classification_report_split_table(split_name, summary)

    return summary

def run_llm_hybrid_augmented(
    llm,
    dataset,
    num_samples: int,
    id2label: Dict[int, str],
    n_shots: int = 0,
    rules: Optional[str] = None,
    importances: Optional[str] = None,
) -> Dict[str, Any]:
    mode = ""
    if rules:
        mode = "(With classical model information)"
    console.rule(f"[bold green] LLM-Based Prediction and Hybrid Modeling {mode}[/bold green]")

    few_shots = None
    train_remaining = dataset.get("train", None)

    if n_shots > 0 and "train" in dataset:
        few_shots, train_remaining = get_few_shot_samples(
            dataset["train"], id2label, n_shots=n_shots, seed=42
        )

    results_by_split: Dict[str, Any] = {}

    for split_name, split_ds in dataset.items():
        console.rule(f"[bold cyan]Split: {split_name}[/bold cyan]")

        data = train_remaining if (split_name == "train" and train_remaining is not None) else split_ds

        summary = evaluate_data_split_with_llm(
            llm=llm,
            split_name=split_name,
            data=data,
            num_samples=num_samples,
            id2label=id2label,
            few_shots=few_shots,
            rules=rules,
            importances=importances,
            show_table=True,
        )

        summary["metadata"]["n_shots"] = n_shots
        results_by_split[split_name] = summary

    return results_by_split


if __name__ == "__main__":
    #llm = LLMClient(host="localhost", port=11434, model_name="mistral:latest") # with ollama
    llm = LLMClient(host="localhost", port=1234, model_name="mistralai_devstral-small-2-24b-instruct-2512") # LM Studio
    #llm = LLMClient(host="172.18.11.146", port=8080, model_name="mistralai/Devstral-Small-2-24B-Instruct-2512") # Server
    llm.test_connection()
    iris, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_iris(random_state=42, test_size=0.1, val_size=0.111)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    config_path = ROOT_DIR / "task2-config.json"
    cfg = load_config(config_path)
    dsA = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test, representation_type="A", print_examples=False)
    dsB = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test, representation_type="B", print_examples=False)

    num_labels = int(cfg["num_labels"])
    print_config_summary(cfg,num_labels)
    id2label, label2id = build_label_maps(cfg=cfg, num_labels=num_labels)

    num_samples = 10
    llmResultsNoRules = run_llm_hybrid_augmented(llm, dsB,  num_samples, id2label, n_shots= 3, rules=None, importances=None)
    llm_summary = {"llm_predictions": {"llm_without_rules": llmResultsNoRules}}
    save_results("output/task2/llm_summary.json", results=llm_summary, cfg=None)


