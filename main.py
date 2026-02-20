from pathlib import Path

from src.task2.decisionTree import run_decision_tree
from src.task2 import transformer
from src.llm import Client as LLMClient
from src.task2.llm_hybrid_augmented import run_llm_hybrid_augmented
from src.task2.helper import load_and_split_iris, get_dataset, save_results, comparative_analysis
from src.utils import build_label_maps, load_config, print_config_summary


from rich.console import Console
console = Console()

def main():
    print("Hello from from-classical-ml-to-llms-in-nlp!")

def task1():
    pass


def task2(llm, random_state=42, test_size=0.1, val_size=0.111, num_samples=10, n_shots= 3, root_dir=Path(""), config_path = "task2-config.json"):
    task2_dir = root_dir / "output" / "task2"
    iris, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_iris(random_state=random_state, test_size=test_size, val_size=val_size)
    feature_names = list(getattr(iris, "feature_names", ["sepal length", "sepal width", "petal length", "petal width"]))
    target_names = list(getattr(iris, "target_names", ["setosa", "versicolor", "virginica"]))

    # Classical Baseline on Structured Data
    clf, metrics, importances, rules = run_decision_tree(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        max_depth=3,
        random_state=42,
        feature_names=feature_names,
        target_names=target_names,
        save_rules_path="output/task2/decision_tree_rules.txt",
        save_plot_path="output/task2/decision_tree.png",
        show_rules=True,
    )
    clf_results = {
        "decision_tree": {
            "train": metrics["train"],
            "val": metrics["val"],
            "test": metrics["test"],
            "importances": importances.to_dict(),
        }
    }

    # Textification and Transformer-Based Modeling
    console.rule("[bold green] Textification and Transformer-Based Modeling [/bold green]")
    cfg = load_config(config_path)
    dsA = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test, representation_type="A", print_examples=False)
    dsB = get_dataset(X_train, y_train, X_val, y_val, X_test, y_test, representation_type="B", print_examples=False)

    num_labels = int(cfg["num_labels"])
    print_config_summary(cfg,num_labels)
    id2label, label2id = build_label_maps(cfg=cfg, num_labels=num_labels)
    tf_results, tf_test_preds = transformer.train_evaluate_experiments(dsA, dsB, id2label, label2id, cfg)
    save_results(f"{task2_dir}/transformer_summary.json", results=tf_results, cfg=cfg)
    tf_results = tf_results["transformer"]["results"]


    # LLM-Based Prediction and Hybrid Modeling
    importances = "\n".join([f"{idx} : {val:.4f}" for idx, val in importances.items()])
    llmTrainResultsNoRules = run_llm_hybrid_augmented(llm=llm, dataset=dsB, num_samples=num_samples, id2label=id2label, n_shots= n_shots, rules=None, importances=None)
    llmTrainResultsWithRules = run_llm_hybrid_augmented(llm=llm, dataset=dsB,  num_samples=num_samples, id2label=id2label, n_shots= n_shots, rules=rules, importances=importances)
    llm_results = {"llm_predictions": {"llm_without_rules": llmTrainResultsNoRules, "llm_with_rules": llmTrainResultsWithRules}}
    save_results("output/task2/llm_summary.json", results=llm_results, cfg=None)

    # Final Comparative Analysis
    comparative_analysis(clf=clf, id2label=id2label, clf_results=clf_results, tf_results=tf_results, tf_test_preds=tf_test_preds, llm_results=llm_results)


if __name__ == "__main__":
    main()
    root_dir = Path(__file__).resolve().parent

    task1()
    llm = LLMClient(host="localhost", port=1234, model_name="mistralai_devstral-small-2-24b-instruct-2512", system_role="developer") # LMStudio # developer role
    #llm = LLMClient(base_url="https://chat-ai.academiccloud.de/v1" , model_name="mistral-large-3-675b-instruct-2512") # system role
    llm.test_connection()
    num_samples = 111 # set to 111 (len(train)).
    task2(llm, random_state=42, test_size=0.1, val_size=0.111, num_samples=num_samples, n_shots= 3, root_dir=root_dir, config_path ="configs/task2-config.json")
