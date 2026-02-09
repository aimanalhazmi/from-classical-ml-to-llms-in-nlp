from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.task2.helper import load_and_split_iris

console = Console()


def run_decision_tree(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    max_depth: int,
    random_state: int,
    feature_names,
    target_names,
    save_rules_path: str | None = None,
    save_plot_path: str | None = None,
    show_rules: bool = True,
):
    console.rule("[bold green]Decision Tree (Classical Model)[/bold green]")

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)

    # preds
    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)

    # metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_f1 = f1_score(y_train, y_pred_train, average="macro")
    val_f1 = f1_score(y_val, y_pred_val, average="macro")
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    console.print(
        Panel.fit(
            f"[bold]Model[/bold]: DecisionTreeClassifier\n"
            f"[bold]max_depth[/bold]={max_depth}, [bold]random_state[/bold]={random_state}",
            title="Summary",
            border_style="green",
        )
    )

    # Metrics table
    t = Table(title="Metrics", show_lines=True)
    t.add_column("Split", style="bold")
    t.add_column("accuracy")
    t.add_column("macro_f1")
    t.add_row("train", f"{train_acc:.4f}", f"{train_f1:.4f}")
    t.add_row("val", f"{val_acc:.4f}", f"{val_f1:.4f}")
    t.add_row("test", f"{test_acc:.4f}", f"{test_f1:.4f}")
    console.print(t)

    # Confusion matrix (test)
    cm = confusion_matrix(y_test, y_pred_test)
    cm_table = Table(title="Confusion Matrix (test)", show_lines=True)
    cm_table.add_column("True \\ Pred", style="bold")
    for name in target_names:
        cm_table.add_column(str(name))
    for i, row in enumerate(cm):
        cm_table.add_row(str(target_names[i]), *[str(int(v)) for v in row])
    console.print(cm_table)

    # Classification report (test)
    report_dict = classification_report(
        y_test, y_pred_test, target_names=target_names, output_dict=True, zero_division=0
    )
    rep_table = Table(title="Classification Report (test)", show_lines=True)
    rep_table.add_column("Class", style="bold")
    rep_table.add_column("precision")
    rep_table.add_column("recall")
    rep_table.add_column("f1-score")
    rep_table.add_column("support")

    for cls_name in target_names:
        row = report_dict[cls_name]
        rep_table.add_row(
            str(cls_name),
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1-score']:.4f}",
            f"{row['support']:.0f}",
        )
    macro = report_dict["macro avg"]
    rep_table.add_row("—", "—", "—", "—", "—")
    rep_table.add_row("macro avg", f"{macro['precision']:.4f}", f"{macro['recall']:.4f}", f"{macro['f1-score']:.4f}", f"{macro['support']:.0f}")
    console.print(rep_table)

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
    imp_table = Table(title="Feature Importance", show_lines=True)
    imp_table.add_column("Feature", style="bold")
    imp_table.add_column("Importance")
    for feat, val in importances.items():
        imp_table.add_row(str(feat), f"{val:.6f}")
    console.print(imp_table)

    # Decision rules
    rules = export_text(clf, feature_names=list(feature_names))
    if show_rules:
        console.print(Panel(rules, title="Decision Rules", border_style="cyan"))

    if save_rules_path:
        Path(save_rules_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_rules_path).write_text(rules, encoding="utf-8")
        console.print(f"[green]Saved rules to:[/green] {save_rules_path}")

    # Plot tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=target_names,
        impurity=True,
    )
    plt.title("Decision Tree Logic")
    plt.tight_layout()
    if save_plot_path:
        Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_plot_path, dpi=200)
        console.print(f"[green]Saved tree plot to:[/green] {save_plot_path}")

    return clf, {"train": {"accuracy": train_acc, "macro_f1": train_f1},
                 "val": {"accuracy": val_acc, "macro_f1": val_f1},
                 "test": {"accuracy": test_acc, "macro_f1": test_f1}}, importances, rules


if __name__ == "__main__":
    iris, X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_iris(
        random_state=42, test_size=0.1, val_size=0.111
    )
    feature_names = list(getattr(iris, "feature_names", ["sepal length", "sepal width", "petal length", "petal width"]))
    target_names = list(getattr(iris, "target_names", ["setosa", "versicolor", "virginica"]))
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
