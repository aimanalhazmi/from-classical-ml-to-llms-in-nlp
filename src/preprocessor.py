import re
import string
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords

from datasets import DatasetDict, Dataset


# formatting imports
from rich.console import Console
from rich.table import Table

# Setup Logger
logger = logging.getLogger("preprocessor")
console = Console()

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model 'en_core_web_sm' not found. Downloading...")
    from spacy.cli import download

    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


@dataclass
class CleanConfig:
    # column & behavior
    create_new_column: bool = True
    output_prefix: str = "cleaned_"

    # row filtering
    drop_empty_rows: bool = True

    # text transforms
    lowercase: bool = True
    replace_br_tags: bool = True  # "<br /><br />" -> "."
    remove_html: bool = True  # remove any tags like <p>...</p>
    remove_mentions: bool = False  # @user
    remove_punctuation: bool = True
    remove_special_chars: bool = False  # removes non-word chars
    normalize_whitespace: bool = True

    # token-level transforms
    remove_stopwords: bool = True
    lemmatize: bool = False
    remove_numbers: bool = False  # removes numeric tokens
    remove_tokens_with_digits: bool = False

    # dataset-level transforms
    deduplicate: bool = True
    dedupe_keep: str = "first"

    # label processing
    encode_labels: bool = False
    label_map: Optional[Dict[str, int]] = None


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text).strip()


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def drop_empty(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df[
        df[col].notna()
        & ~df[col].astype(str).str.strip().eq("")
        & ~df[col].astype(str).str.strip().eq("` `")
        ]


def remove_stopwords(x: str) -> str:
    return " ".join([w for w in x.split() if w not in STOP_WORDS])


def token_filter(doc, config: CleanConfig) -> str:
    toks = []
    for t in doc:
        if config.remove_numbers and t.like_num:
            continue
        if config.remove_tokens_with_digits and any(ch.isdigit() for ch in t.text):
            continue
        toks.append(t.lemma_ if config.lemmatize else t.text)
    return " ".join(toks)


def apply_preprocessing_to_splits(dataset_dict, config_obj, text_column, label_column, force_logging=False):
    cleaned_splits = {}
    processed_col_name = text_column

    console.rule("Starting preprocessing")
    for split_name, ds in dataset_dict.items():
        logger.info(f"Preprocessing split: {split_name}...")

        df = ds.to_pandas()

        df_clean, new_col_name = preprocess_all(
            df,
            text_column=text_column,
            label_column=label_column,
            config=config_obj
        )
        processed_col_name = new_col_name

        cleaned_ds = Dataset.from_pandas(df_clean)

        if "labels" in cleaned_ds.column_names:
            original_features = ds.features["labels"]
            cleaned_ds = cleaned_ds.cast_column("labels", original_features)

        cleaned_splits[split_name] = cleaned_ds

    return DatasetDict(cleaned_splits), processed_col_name

def preprocess_all(
        df: pd.DataFrame,
        text_column: str,
        label_column: Optional[str] = None,
        config: Optional[CleanConfig] = None,
        *,
        subset_for_dedupe: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Single entry-point preprocessing function.

    Returns:
        (processed_df, processed_text_column_name)
    """
    if config is None:
        config = CleanConfig()

    logger.info(f"Starting preprocessing on column: '{text_column}'")

    if text_column not in df.columns:
        logger.error(f"Column '{text_column}' missing from DataFrame.")
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    out_df = df.copy()
    original_rows = len(out_df)

    # drop empty rows
    if config.drop_empty_rows:
        out_df[text_column] = out_df[text_column].astype(str)
        out_df = drop_empty(out_df, text_column)

    # output column
    processed_col = (
        f"{config.output_prefix}{text_column}"
        if config.create_new_column
        else text_column
    )

    logger.info("Applying string transformations (lowercase, html, punctuation)...")
    series = out_df[text_column].astype(str)

    if config.lowercase:
        series = series.str.lower()

    if config.replace_br_tags:
        series = series.str.replace("<br /><br />", ".", regex=False)

    if config.remove_html:
        series = series.apply(strip_html)

    if config.remove_mentions:
        series = series.apply(remove_mentions)

    if config.remove_punctuation:
        series = series.apply(lambda x: x.translate(PUNCT_TABLE))

    if config.remove_special_chars:
        series = series.str.replace(r"[^\w\s]", "", regex=True)

    if config.normalize_whitespace:
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()

    # spaCy/token ops
    if config.lemmatize or config.remove_numbers or config.remove_tokens_with_digits:
        logger.info("Running spaCy pipeline (lemmatization/token filtering)... this may take a moment.")
        docs = nlp.pipe(series.tolist(), batch_size=512)
        series = pd.Series([token_filter(d, config) for d in docs], index=out_df.index)

    if config.remove_stopwords:
        logger.info("Removing stopwords...")
        series = series.apply(remove_stopwords)

    if config.normalize_whitespace:
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()

    out_df[processed_col] = series

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

        # Count before drop
        duplicated_count = out_df.duplicated(subset=subset, keep=config.dedupe_keep).sum()
        out_df = out_df.drop_duplicates(subset=subset, keep=config.dedupe_keep)

    # label encoding
    if config.encode_labels:
        if label_column is None:
            raise ValueError("encode_labels=True requires label_column.")

        logger.info(f"Encoding labels in column: '{label_column}'")
        mapping = config.label_map or {"positive": 1, "negative": 0}
        out_df[label_column] = out_df[label_column].map(mapping)

    # summary table
    table = Table(title="Preprocessing Summary", style="cyan")

    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right")

    table.add_row("Initial Rows", str(original_rows))
    table.add_row("Rows Removed (Empty)", str(original_rows - len(out_df) - duplicated_count))
    table.add_row("Rows Removed (Duplicate)", str(duplicated_count))
    table.add_row("Final Rows", str(len(out_df)))
    table.add_row("Processed Column", processed_col)

    if config.encode_labels and label_column:
        table.add_row("Label Encoding", "Applied")

    console.print(table)
    logger.info("Preprocessing complete.")

    return out_df, processed_col