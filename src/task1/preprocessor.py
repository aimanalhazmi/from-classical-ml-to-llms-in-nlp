import re
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from collections import Counter


import pandas as pd
import emoji
from sklearn.model_selection import train_test_split

from src.utils import get_project_root
# formatting
from rich.console import Console
from rich.table import Table

# Setup Logger
logger = logging.getLogger("preprocessor")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()


@dataclass
class SMM4HConfig:
    """
    Configuration specifically tuned for SMM4H Medical NLP Task.
    """
    # column & behavior
    create_new_column: bool = False
    output_prefix: str = "clean_"

    # row filtering
    drop_empty_rows: bool = True

    # text transforms
    lowercase: bool = False  # Keep False for Medical NER/BERT-cased
    replace_br_tags: bool = True  # "<br />" -> ". "
    remove_html: bool = True  # <p> -> " "

    anonymize_mentions: bool = True  # @user -> [USER]
    anonymize_urls: bool = True  # http://... -> [URL]
    demojize: bool = True  # 💊 -> :pill:
    fix_hashtags: bool = True  # #flu -> flu

    # Cleaning
    normalize_whitespace: bool = True

    # dataset-level transforms
    deduplicate: bool = True
    dedupe_keep: str = "first"

    # label processing
    encode_labels: bool = False
    label_map: Optional[Dict[str, int]] = None


def process_smm4h_text(text: str, config: SMM4HConfig, stats: Counter) -> str:
    """
    Custom pipeline for Social Media Medical data.
    """
    if not isinstance(text, str):
        return str(text)

    original_text = text

    # HTML & Breaks
    if config.replace_br_tags:
        if "<br" in text:
            stats['HTML Removed'] += 1
        text = text.replace("<br />", ". ").replace("<br>", ". ")

    if config.remove_html:
        text, n = re.subn(r"<[^>]+>", " ", text)
        if n > 0: stats['HTML Tags Removed'] += n

    # Anonymization
    if config.anonymize_mentions:
        text, n = re.subn(r'@[\w_]+', '[USER]', text)
        if n > 0: stats['User Mentions Anonymized'] += 1  # Count records, roughly

    if config.anonymize_urls:
        text, n = re.subn(r'(?:https?://|HTTPURL)\S+', '[URL]', text, flags=re.IGNORECASE)
        if n > 0: stats['URLs Anonymized'] += 1

    # Social Media Cleaning
    if config.demojize:
        # Converts emojis to text (🤢 -> :nauseated_face:)
        demojized = emoji.demojize(text, delimiters=(" :", ": "))
        deduplicated = re.sub(r'(:[a-z0-9_+-]+:)(?:\s*\1)+', r'\1', demojized)
        if deduplicated != text:
            stats['Emojis Converted/Deduped'] += 1
        text = deduplicated

    if config.fix_hashtags:
        if '#' in text:
            stats['Hashtags Cleaned'] += 1
            text = text.replace('#', '')

    if config.lowercase:
        text = text.lower()

    if config.normalize_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def drop_empty(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[
        df[col].notna()
        & ~df[col].astype(str).str.strip().eq("")
        & ~df[col].astype(str).str.strip().eq("` `")
        ]


def preprocess(
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        config: Optional[SMM4HConfig] = None,
        *,
        subset_for_dedupe: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Single entry-point preprocessing function.
    Returns: (processed_df, processed_text_column_name)
    """
    if config is None:
        config = SMM4HConfig()

    logger.info(f"Starting preprocessing on column: '{text_column}'")

    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' missing from DataFrame.")
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    out_df = df.copy()
    original_rows = len(out_df)
    stats = Counter()

    # drop empty rows first
    if config.drop_empty_rows:
        out_df[text_column] = out_df[text_column].astype(str)
        out_df = drop_empty(out_df, text_column)

    # output column
    if config.create_new_column:
        processed_col = f"{config.output_prefix}{text_column}"
    else:
        out_df["text_before"] = out_df[text_column]
        processed_col = text_column

    logger.info("Applying SMM4H string transformations...")

    # Apply pipeline with stats tracking
    out_df[processed_col] = out_df[text_column].astype(str).apply(
        lambda x: process_smm4h_text(x, config, stats)
    )

    if config.drop_empty_rows:
        out_df = drop_empty(out_df, processed_col)

    # dedupe
    duplicated_count = 0
    if config.deduplicate:
        logger.info("Checking for duplicates...")
        if subset_for_dedupe is not None:
            subset = subset_for_dedupe
        else:
            subset = [processed_col] + ([label_column] if label_column else [])

        duplicated_count = out_df.duplicated(subset=subset, keep=config.dedupe_keep).sum()
        out_df = out_df.drop_duplicates(subset=subset, keep=config.dedupe_keep)

    if label_column and label_column in out_df.columns:
        logger.info(f"Fixing label types in '{label_column}'...")
        # Drop rows where label is NaN (cannot convert NaN to int)
        before_label_drop = len(out_df)
        out_df = out_df.dropna(subset=[label_column])
        if len(out_df) < before_label_drop:
            logger.warning(f"Dropped {before_label_drop - len(out_df)} rows with missing labels.")

        # Convert to Integer
        out_df[label_column] = out_df[label_column].astype(int)

    # summary table
    table = Table(title="SMM4H Preprocessing Summary", style="cyan")

    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right")

    table.add_row("Initial Rows", str(original_rows))
    table.add_row("Rows Removed (Empty)", str(original_rows - len(out_df) - duplicated_count))
    table.add_row("Rows Removed (Duplicate)", str(duplicated_count))
    table.add_row("Final Rows", str(len(out_df)))
    table.add_section()

    # Add step-specific stats
    for k, v in stats.items():
        table.add_row(f"Transformed: {k}", str(v))

    console.print(table)
    logger.info("Preprocessing complete.")

    return out_df, processed_col


def print_data_stats(df: pd.DataFrame, dataset_name: str):
    """
    Prints a Rich table showing counts per language and label distribution.
    """
    table = Table(title=f"Statistics: {dataset_name}", style="bold green")
    table.add_column("Language", style="cyan")
    table.add_column("Total Rows", justify="right")
    table.add_column("Label 0 (Neg)", justify="right")
    table.add_column("Label 1 (Pos)", justify="right")
    table.add_column("Pos Ratio", justify="right")

    for lang in df['language'].unique():
        subset = df[df['language'] == lang]
        total = len(subset)
        pos = len(subset[subset['label'] == 1])
        neg = len(subset[subset['label'] == 0])
        ratio = f"{(pos / total) * 100:.1f}%" if total > 0 else "0%"

        table.add_row(lang, str(total), str(neg), str(pos), ratio)

    # Total row
    total = len(df)
    pos = len(df[df['label'] == 1])
    neg = len(df[df['label'] == 0])
    ratio = f"{(pos / total) * 100:.1f}%" if total > 0 else "0%"
    table.add_section()
    table.add_row("TOTAL", str(total), str(neg), str(pos), ratio)

    console.print(table)

if __name__ == "__main__":
    project_dir = get_project_root()
    data_dir = project_dir / "data/SMM4H_2026"
    train_data_path = data_dir / "train_data_SMM4H_2026_Task_1.csv"
    dev_data_path = data_dir / "dev_data_SMM4H_2026_Task_1.csv"

    try:
        logger.info(f"Loading raw data...")
        train_data = pd.read_csv(train_data_path)
        dev_data = pd.read_csv(dev_data_path)
    except FileNotFoundError as e:
        logger.error(f"Could not find file: {e}")
        exit(1)

    # SELECT LANGUAGES
    train_langs = ["en", "ru"]  # English and Russian for Training
    zero_shot_lang = "de"  # German for Evaluation
    selected_langs = train_langs + [zero_shot_lang]

    logger.info(f"Filtering for languages: {selected_langs}")
    train_val = train_data[train_data["language"].isin(selected_langs)].copy()
    test = dev_data[dev_data["language"].isin(selected_langs)].copy()

    console.rule("[bold red]Step 1: Data Exploration (Raw)[/bold red]")
    print_data_stats(train_val, "Raw Train/Val Data")
    print_data_stats(test, "Raw Test Data")

    console.rule("[bold red]Step 2: Preprocessing[/bold red]")

    logger.info("Processing Train/Val set...")
    train_val_clean, text_col = preprocess(
        train_val,
        text_column="text",
        label_column="label",
        config=SMM4HConfig()
    )

    logger.info("Processing Test set...")
    test_clean, _ = preprocess(
        test,
        text_column="text",
        label_column="label",
        config=SMM4HConfig()
    )

    console.rule("[bold red]Step 3: Splitting[/bold red]")
    logger.info("Splitting Train/Val into final TRAIN and VAL sets (80/20)...")

    # Create a helper column for Stratification (Language + Label)
    # This ensures every split has balanced labels for EVERY language
    train_val_clean['stratify_col'] = (
            train_val_clean['language'] + "_" + train_val_clean['label'].astype(str)
    )

    train_df, val_df = train_test_split(
        train_val_clean,
        test_size=0.2,
        stratify=train_val_clean['stratify_col'],
        random_state=42
    )
    # Cleanup helper column
    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])

    print_data_stats(train_df, "Final Train Set")
    print_data_stats(val_df, "Final Val Set")

    console.rule("[bold red]Step 4: Saving[/bold red]")
    out_train = data_dir / "train.csv"
    out_val = data_dir / "val.csv"
    out_test = data_dir / "test.csv"

    train_df.to_csv(out_train, index=False)
    val_df.to_csv(out_val, index=False)
    test_clean.to_csv(out_test, index=False)

    logger.info(f"Saved Train ({len(train_df)} rows) to {out_train}")
    logger.info(f"Saved Val   ({len(val_df)} rows) to {out_val}")
    logger.info(f"Saved Test  ({len(test_clean)} rows) to {out_test}")

    console.print(
        "\n[bold yellow]⚠️  NOTE:[/bold yellow] The 'train.csv' file currently contains "
        f"all 3 languages ({selected_langs}).\n"
        "When training for Task 1, remember to [bold]filter out the zero-shot language[/bold] "
        f"('{zero_shot_lang}') from the training set loader!"
    )