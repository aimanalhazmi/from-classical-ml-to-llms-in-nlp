import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from rich.console import Console
from rich.table import Table

from src.utils import get_project_root


logger = logging.getLogger("explorer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
console = Console()

LABEL_COLOR_MAP = {0: "firebrick", 1: "forestgreen"}
LABEL_ORDER = [0, 1]

sns.set_theme(style="whitegrid")


def ensure_output_dir(base_dir: Path, task_name: str = "task1") -> Path:
    out_dir = base_dir / "output" / task_name  / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    files = {
        "train": "train.csv",
        "val": "val.csv",
        "test": "test.csv",
        "translated": "train_de_translated.csv",
    }

    for name, filename in files.items():
        fpath = data_dir / filename
        if not fpath.exists():
            continue

        df = pd.read_csv(fpath)

        if "text" in df.columns:
            df["text"] = df["text"].fillna("").astype(str)

        if name != "translated" and "text" in df.columns and "token_count" not in df.columns:
            df["token_count"] = df["text"].str.split().str.len()

        # normalize type to string if present
        if "type" in df.columns:
            df["type"] = df["type"].fillna("").astype(str)

        # normalize language/label if present
        if "language" in df.columns:
            df["language"] = df["language"].fillna("").astype(str)

        datasets[name] = df
        logger.info(f"Loaded {name}: {len(df)} rows")

    return datasets


def build_type_palette(
    datasets: Dict[str, pd.DataFrame], names: Tuple[str, ...] = ("train", "val", "test")
) -> Tuple[Optional[List[str]], Optional[Dict[str, object]]]:
    all_types: List[str] = []
    for n in names:
        if n not in datasets:
            continue
        df = datasets[n]
        if "type" not in df.columns:
            continue

        vals = df["type"].dropna().astype(str)
        vals = vals[vals.str.len() > 0]
        all_types.extend(vals.tolist())

    type_order = sorted(set(all_types))
    if not type_order:
        return None, None

    colors = sns.color_palette("Set2", n_colors=len(type_order))
    type_palette = dict(zip(type_order, colors))
    return type_order, type_palette


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return s if not s.empty else None


def print_dataset_overview(df: pd.DataFrame, dataset_name: str) -> None:
    """High-level dataset overview: rows, languages, labels, types, missingness."""
    console.rule(f"[bold cyan]Dataset Overview: {dataset_name} ({len(df):,} rows)[/bold cyan]")

    # Missingness snapshot
    cols_to_check = [c for c in ["text", "language", "label", "type", "token_count"] if c in df.columns]
    miss_table = Table(title=f"Missingness ({dataset_name})", show_lines=True)
    miss_table.add_column("Column", style="magenta")
    miss_table.add_column("Missing", justify="right")
    miss_table.add_column("Missing %", justify="right")

    for c in cols_to_check:
        missing = int(df[c].isna().sum())
        miss_table.add_row(c, f"{missing:,}", _pct(missing / len(df) if len(df) else 0.0))

    console.print(miss_table)

    # Quick cardinalities
    card_table = Table(title=f"Cardinality ({dataset_name})", show_lines=True)
    card_table.add_column("Field", style="magenta")
    card_table.add_column("Unique Values", justify="right")
    card_table.add_column("Values", style="white")

    if "language" in df.columns:
        langs = sorted(set(df["language"].dropna().astype(str)))
        card_table.add_row("language", f"{len(langs)}", ", ".join(langs[:12]) + (" ..." if len(langs) > 12 else ""))

    if "label" in df.columns:
        labels = sorted(set(df["label"].dropna().astype(int)))
        card_table.add_row("label", f"{len(labels)}", ", ".join(map(str, labels)))

    if "type" in df.columns:
        types = sorted(set(df["type"].dropna().astype(str)))
        types = [t for t in types if t]
        card_table.add_row("type", f"{len(types)}", ", ".join(types[:12]) + (" ..." if len(types) > 12 else ""))

    console.print(card_table)


def print_length_stats(df: pd.DataFrame, dataset_name: str) -> None:
    if "token_count" not in df.columns:
        return

    # Overall
    s = df["token_count"].dropna()
    if s.empty:
        return

    overall = Table(title=f"Token Length Stats (Overall) — {dataset_name}", show_lines=True)
    overall.add_column("Metric", style="magenta")
    overall.add_column("Value", justify="right")

    overall.add_row("Count", f"{len(s):,}")
    overall.add_row("Mean", f"{s.mean():.2f}")
    overall.add_row("Std", f"{s.std(ddof=1):.2f}" if len(s) > 1 else "0.00")
    overall.add_row("Min", f"{int(s.min())}")
    overall.add_row("P25", f"{np.percentile(s, 25):.0f}")
    overall.add_row("Median", f"{np.percentile(s, 50):.0f}")
    overall.add_row("P75", f"{np.percentile(s, 75):.0f}")
    overall.add_row("P90", f"{np.percentile(s, 90):.0f}")
    overall.add_row("P95", f"{np.percentile(s, 95):.0f}")
    overall.add_row("Max", f"{int(s.max())}")
    console.print(overall)

    if "language" in df.columns:
        lang_table = Table(title=f"Token Length Stats by Language — {dataset_name}", show_lines=True)
        lang_table.add_column("Language", style="cyan")
        lang_table.add_column("Count", justify="right")
        lang_table.add_column("Mean", justify="right")
        lang_table.add_column("Median", justify="right")
        lang_table.add_column("P90", justify="right")
        lang_table.add_column("Max", justify="right")

        for lang in sorted(df["language"].dropna().unique()):
            ss = df.loc[df["language"] == lang, "token_count"].dropna()
            if ss.empty:
                continue
            lang_table.add_row(
                str(lang),
                f"{len(ss):,}",
                f"{ss.mean():.2f}",
                f"{np.percentile(ss, 50):.0f}",
                f"{np.percentile(ss, 90):.0f}",
                f"{int(ss.max())}",
            )

        console.print(lang_table)

    if "label" in df.columns:
        lab_table = Table(title=f"Token Length Stats by Label — {dataset_name}", show_lines=True)
        lab_table.add_column("Label", style="cyan")
        lab_table.add_column("Count", justify="right")
        lab_table.add_column("Mean", justify="right")
        lab_table.add_column("Median", justify="right")
        lab_table.add_column("P90", justify="right")
        lab_table.add_column("Max", justify="right")

        for lab in LABEL_ORDER:
            ss = df.loc[df["label"] == lab, "token_count"].dropna()
            if ss.empty:
                continue
            lab_table.add_row(
                str(lab),
                f"{len(ss):,}",
                f"{ss.mean():.2f}",
                f"{np.percentile(ss, 50):.0f}",
                f"{np.percentile(ss, 90):.0f}",
                f"{int(ss.max())}",
            )

        console.print(lab_table)


def print_distributions(df: pd.DataFrame, dataset_name: str) -> None:
    """Print distribution tables for label and type (and language)."""
    if "label" in df.columns:
        dist = df["label"].value_counts(dropna=False).to_dict()
        t = Table(title=f"Label Distribution — {dataset_name}", show_lines=True)
        t.add_column("Label", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Percent", justify="right")

        total = len(df)
        for lab in LABEL_ORDER:
            cnt = int(dist.get(lab, 0))
            t.add_row(str(lab), f"{cnt:,}", _pct(cnt / total if total else 0.0))
        # show NaN if any
        nan_cnt = int(dist.get(np.nan, 0)) if any(pd.isna(k) for k in dist.keys()) else int(df["label"].isna().sum())
        if nan_cnt:
            t.add_row("NaN", f"{nan_cnt:,}", _pct(nan_cnt / total if total else 0.0))

        console.print(t)

    # Language distribution
    if "language" in df.columns:
        vc = df["language"].value_counts(dropna=False)
        t = Table(title=f"Language Distribution — {dataset_name}", show_lines=True)
        t.add_column("Language", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Percent", justify="right")

        total = len(df)
        for k, v in vc.items():
            t.add_row(str(k), f"{int(v):,}", _pct(int(v) / total if total else 0.0))
        console.print(t)

    # Type distribution
    if "type" in df.columns:
        vc = df["type"].value_counts(dropna=False)
        t = Table(title=f"Source Type Distribution — {dataset_name}", show_lines=True)
        t.add_column("Type", style="cyan")
        t.add_column("Count", justify="right")
        t.add_column("Percent", justify="right")

        total = len(df)
        for k, v in vc.items():
            t.add_row(str(k), f"{int(v):,}", _pct(int(v) / total if total else 0.0))
        console.print(t)


def print_label_by_language(df: pd.DataFrame, dataset_name: str, label_col: str = "label") -> None:
    if "language" not in df.columns or label_col not in df.columns:
        return

    ct = pd.crosstab(df["language"], df[label_col], normalize=False)

    for lab in LABEL_ORDER:
        if lab not in ct.columns:
            ct[lab] = 0
    ct = ct[LABEL_ORDER]

    t = Table(title=f"Label by Language (Counts) — {dataset_name}", show_lines=True)
    t.add_column("Language", style="cyan")
    t.add_column("Negative", justify="right")
    t.add_column("positive", justify="right")
    t.add_column("Total", justify="right")
    t.add_column("Pos %", justify="right")

    for lang in ct.index.astype(str).tolist():
        c0 = int(ct.loc[lang, 0])
        c1 = int(ct.loc[lang, 1])
        tot = c0 + c1
        pos_pct = (c1 / tot) if tot else 0.0
        t.add_row(lang, f"{c0:,}", f"{c1:,}", f"{tot:,}", _pct(pos_pct))

    console.print(t)


def print_text_quality_checks(df: pd.DataFrame, dataset_name: str) -> None:
    """Basic QA: empty texts, very short/very long outliers."""
    if "text" not in df.columns:
        return

    texts = df["text"].fillna("").astype(str)
    empty = int((texts.str.strip().str.len() == 0).sum())
    total = len(df)

    if "token_count" in df.columns:
        lengths = df["token_count"].fillna(0).astype(int)
        metric = "tokens"
    else:
        lengths = texts.str.len()
        metric = "chars"

    short_0_3 = int((lengths <= 3).sum())
    long_p99 = int(np.percentile(lengths, 99)) if len(lengths) else 0
    very_long = int((lengths >= long_p99).sum()) if long_p99 else 0

    t = Table(title=f"Text QA Checks — {dataset_name}", show_lines=True)
    t.add_column("Check", style="magenta")
    t.add_column("Count", justify="right")
    t.add_column("Percent", justify="right")

    t.add_row("Empty/blank text", f"{empty:,}", _pct(empty / total if total else 0.0))
    t.add_row(f"Very short ({metric} ≤ 3)", f"{short_0_3:,}", _pct(short_0_3 / total if total else 0.0))
    if long_p99:
        t.add_row(f"Very long ({metric} ≥ P99={long_p99})", f"{very_long:,}", _pct(very_long / total if total else 0.0))

    console.print(t)

def plot_label_distribution(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    if "language" not in df.columns or "label" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df,
        x="language",
        hue="label",
        hue_order=LABEL_ORDER,
        palette=LABEL_COLOR_MAP,
        edgecolor="black",
    )

    ax.set_title(f"Label Distribution ({dataset_name})", fontsize=14)
    ax.set_xlabel("Language")
    ax.set_ylabel("Count")
    ax.legend(title="Label", labels=["Negative (0)", "Positive (1)"])
    sns.despine()

    for container in ax.containers:
        ax.bar_label(container, padding=2)

    plt.tight_layout()
    plt.savefig(out_dir / f"dist_{dataset_name}.png", dpi=300)
    plt.close()


def plot_type_distribution(
    df: pd.DataFrame,
    dataset_name: str,
    out_dir: Path,
    type_order: Optional[List[str]] = None,
    type_palette: Optional[Dict[str, object]] = None,
) -> None:
    if "type" not in df.columns or not type_order or not type_palette:
        return

    df_plot = df.copy()
    df_plot["type"] = df_plot["type"].fillna("").astype(str)
    df_plot = df_plot[df_plot["type"].isin(type_order)]
    if df_plot.empty:
        return

    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        data=df_plot,
        x="type",
        order=type_order,
        hue="type",
        hue_order=type_order,
        palette=type_palette,
        edgecolor="black",
        legend=False,
    )

    ax.set_title(f"Source Type Distribution ({dataset_name})", fontsize=14)
    ax.set_xlabel("Source Type")
    ax.set_ylabel("Count")
    sns.despine()

    for container in ax.containers:
        ax.bar_label(container, padding=2)

    plt.tight_layout()
    plt.savefig(out_dir / f"type_dist_{dataset_name}.png", dpi=300)
    plt.close()


def plot_sentence_lengths(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    if "language" not in df.columns or "token_count" not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=df,
        x="language",
        y="token_count",
        hue="language",
        palette="Pastel1",
        legend=False,
    )

    ax = plt.gca()
    ax.set_title(f"Sentence Length Distribution ({dataset_name})", fontsize=14)
    ax.set_xlabel("Language")
    ax.set_ylabel("Tokens")
    sns.despine()

    plt.tight_layout()
    plt.savefig(out_dir / f"length_{dataset_name}.png", dpi=300)
    plt.close()


def plot_top_ngrams(
    df: pd.DataFrame,
    dataset_name: str,
    out_dir: Path,
    n: int = 2,
    top_k: int = 10,
) -> None:
    if "language" not in df.columns or "label" not in df.columns or "text" not in df.columns:
        return

    for lang in df["language"].dropna().unique():
        lang_subset = df[df["language"] == lang]

        for label in lang_subset["label"].dropna().unique():
            subset = lang_subset[lang_subset["label"] == label]
            if len(subset) < 5:
                continue

            try:
                vec = CountVectorizer(
                    ngram_range=(n, n),
                    max_features=top_k,
                    stop_words="english" if str(lang).lower() == "en" else None,
                ).fit(subset["text"])

                bag = vec.transform(subset["text"])
                sum_words = bag.sum(axis=0)

                words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
                words_freq.sort(key=lambda x: x[1], reverse=True)
                if not words_freq:
                    continue

                ngram_df = pd.DataFrame(words_freq, columns=["Ngram", "Frequency"])

                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x="Frequency",
                    y="Ngram",
                    data=ngram_df,
                    hue="Ngram",
                    palette="viridis",
                    legend=False,
                )

                label_name = "Positive" if int(label) == 1 else "Negative"
                plt.title(f"Top {n}-grams: {str(lang).upper()} - {label_name} ({dataset_name})", fontsize=14)
                sns.despine()
                plt.tight_layout()

                plt.savefig(out_dir / f"ngram_{dataset_name}_{lang}_{label_name}.png", dpi=300)
                plt.close()

            except Exception as e:
                logger.warning(f"Skipping N-gram plot for {lang} label {label}: {e}")


def generate_wordcloud(corpus: str, lang: str, dataset_name: str, out_dir: Path) -> None:
    if not corpus.strip():
        return

    try:
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=100,
        ).generate(corpus)

        wc.to_file(out_dir / f"wordcloud_{dataset_name}_{lang}.png")
    except Exception as e:
        logger.warning(f"WordCloud failed for {lang}: {e}")


def manual_inspection(df: pd.DataFrame, n: int = 2) -> None:
    if "language" not in df.columns or "label" not in df.columns or "text" not in df.columns:
        return

    console.rule(f"[bold red]Manual Inspection ({len(df)} samples)[/bold red]")

    for lang in df["language"].dropna().unique():
        subset = df[df["language"] == lang]
        pos = subset[subset["label"] == 1]
        neg = subset[subset["label"] == 0]

        if len(pos) < n or len(neg) < n:
            continue

        t = Table(show_header=True, header_style="bold yellow", title=f"Samples: {lang}")
        t.add_column("Positive (ADE)", style="green")
        t.add_column("Negative (No ADE)", style="red")

        p_samples = pos["text"].sample(n, random_state=42).tolist()
        n_samples = neg["text"].sample(n, random_state=42).tolist()

        for p, ng in zip(p_samples, n_samples):
            t.add_row(p, ng)
            t.add_section()

        console.print(t)


def compare_translation(df_trans: pd.DataFrame, n: int = 3) -> None:
    console.rule("[bold magenta]Translation Quality Check[/bold magenta]")

    if "text_source" not in df_trans.columns or "text" not in df_trans.columns:
        return

    table = Table(title="English vs. Machine Translated German", show_lines=True)
    table.add_column("Original (EN)", style="cyan", ratio=1)
    table.add_column("Translated (DE)", style="magenta", ratio=1)

    for _, row in df_trans.head(n).iterrows():
        table.add_row(str(row["text_source"]), str(row["text"]))

    console.print(table)


def run_analysis(datasets: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    type_order, type_palette = build_type_palette(datasets)

    for name in ("train", "val", "test"):
        if name not in datasets:
            continue

        df = datasets[name]
        logger.info(f"Analyzing {name}...")

        # Console stats (tables)
        print_dataset_overview(df, name)
        print_distributions(df, name)
        print_label_by_language(df, name)
        print_length_stats(df, name)
        print_text_quality_checks(df, name)

        # Plots
        plot_label_distribution(df, name, out_dir)
        plot_type_distribution(df, name, out_dir, type_order=type_order, type_palette=type_palette)
        plot_sentence_lengths(df, name, out_dir)

        if name in ("train", "test"):
            plot_top_ngrams(df, name, out_dir, n=2, top_k=10)

        if "language" in df.columns and "text" in df.columns:
            for lang in df["language"].dropna().unique():
                subset = df[df["language"] == lang]
                corpus = " ".join(subset["text"].tolist())
                generate_wordcloud(corpus, str(lang), name, out_dir)

        manual_inspection(df)

    if "translated" in datasets:
        compare_translation(datasets["translated"])


if __name__ == "__main__":
    project_dir = get_project_root()
    data_dir = project_dir / "data/SMM4H_2026"
    out_dir = ensure_output_dir(project_dir, task_name="task1")

    data = load_data(data_dir)
    run_analysis(data, out_dir)

    logger.info(f"Analysis Complete. Plots saved to '{out_dir}'")
