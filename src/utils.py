from pathlib import Path
import json
import pandas as pd
from rich.console import Console
from rich.pretty import Pretty
from rich.table import Table
from rich.panel import Panel

console = Console()

def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def print_config_summary(cfg: dict, num_labels: int):
    id2label, label2id = build_label_maps(cfg, num_labels)

    console.print(Panel.fit("Loaded configuration", style="bold"))
    console.print(Pretty(cfg))
    t = Table(title="Label mappings", show_lines=True)
    t.add_column("id", style="bold")
    t.add_column("label")
    for i in sorted(id2label.keys()):
        t.add_row(str(i), id2label[i])
    console.print(t)
    console.print(Panel.fit("label2id", style="bold"))
    console.print(Pretty(label2id))

def build_label_maps(cfg: dict, num_labels: int):
    labels = cfg.get("labels")
    if labels is None:
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}
        return id2label, label2id
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError('`labels` in config must be a list of strings')
    if len(labels) != num_labels:
        raise ValueError(f"len(labels)={len(labels)} but num_labels={num_labels}")
    id2label = {i: name for i, name in enumerate(labels)}
    label2id = {name: i for i, name in enumerate(labels)}
    return id2label, label2id


def get_project_root() -> Path:
    """
    Returns project root folder.
    """
    current_path = Path(__file__).resolve()
    for parent in [current_path] + list(current_path.parents):
        if (parent / '.git').exists() or (parent / 'pyproject.toml').exists():
            return parent
    return current_path.parent


def display_samples(df: pd.DataFrame, n: int = 5):
    """Displays a Rich Table with original and translated examples."""
    table = Table(title=f"Translation Preview (First {n} samples)", show_lines=True)
    table.add_column("Original (EN)", style="cyan", ratio=1)
    table.add_column("Translated (DE)", style="green", ratio=1)

    # Take the first n successfully translated rows
    preview_df = df.head(n)

    for _, row in preview_df.iterrows():
        table.add_row(str(row['text_source']), str(row['text']))

    console.print(table)