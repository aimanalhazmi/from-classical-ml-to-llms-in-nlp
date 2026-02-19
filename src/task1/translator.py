import logging
import time
from pathlib import Path
import pandas as pd

from src.utils import get_project_root, display_samples
from src.llm import Client

# Formatting
from tqdm.auto import tqdm
from rich.console import Console
from rich.panel import Panel

# Setup Logger
logger = logging.getLogger("translator")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

project_root = get_project_root()

# --- CONFIGURATION ---

INPUT_FILE = project_root / "data/SMM4H_2026/train.csv"
OUTPUT_FILE =  project_root / "data/SMM4H_2026/train_de_translated.csv"
SAMPLE_SIZE =  1000
POS_FRAC = 0.08
SOURCE_LANG = "English"
TARGET_LANG = "German"
SOURCE_LANG_COL = "en"
TARGET_LANG_COL = "de"

HOST =  "localhost" #"172.18.11.146"
PORT = 1234 # 8080
BASE_URL = "https://chat-ai.academiccloud.de/v1"
MODEL_NAME = "mistralai_devstral-small-2-24b-instruct-2512" # "llama-3.1-sauerkrautlm-70b-instruct"  # "mistralai/Devstral-Small-2-24B-Instruct-2512"
SYSTEM_ROLE = "developer" # system developer
BATCH_SIZE = 25
BATCH_PAUSE_SECONDS = 60
MAX_RETRIES = 6
BACKOFF_BASE_SECONDS = 2
CHECKPOINT_EVERY = 50

SYSTEM_MESSAGE = (
    "You are a professional medical translator. "
    f"Your task is to translate the following {SOURCE_LANG} social media post into {TARGET_LANG}.\n"
    "### Rules ###:\n"
    "1. Preserve the original medical meaning exactly.\n"
    "2. Keep medical acronyms (e.g., 'PFS', 'AIDS') in their original form if commonly used in {TARGET_LANG} medical context.\n"
    "3. DO NOT: explain, comment, add prefaces (e.g., 'Translation:'), add notes, add quotes, add extra lines.\n"
    "4. Do not add any conversational filler (e.g., 'Here is the translation'). Return ONLY the {TARGET_LANG} translation."
    "5. f input is empty, output empty.\n"
)


def formatTranslationMessage( text: str, source_lang: str = "English", target_lang: str = "German") -> str:
    """ Constructs a dynamic prompt for Medical Translation """
    userMessage = (
    f"""Translate the following medical social media post from {source_lang} to {target_lang}.\n
    ### TEXT TO TRANSLATE ###
    {text}
    
    ### INSTRUCTION ###
    Return ONLY the {target_lang} translation.\n"
    Do not add conversational fillers like "Here is the translation".
    Do not wrap the output in quotes.
    """)
    return userMessage


def translate_dataset(df: pd.DataFrame, llm: Client, source_lang="English", target_lang="German"):

    console.rule("[bold blue]Starting Translation Task[/bold blue]")
    logger.info(f"Processing {len(df)} samples...")

    df_result = df.copy()
    df_result['text_source'] = df_result["text"].astype(str)
    df_result["text"] = None
    df_result["origin"] = "de_translated"
    output_file = Path(OUTPUT_FILE)

    # Optional: resume if checkpoint exists
    if output_file.exists():
        try:
            prev = pd.read_csv(output_file)

            if "id" in prev.columns and "text" in prev.columns:
                # map id -> translated text
                prev_map = prev.dropna(subset=["text"]).set_index("id")["text"]
                df_result["text"] = df_result["id"].map(prev_map)
                logger.info(f"Resumed {df_result['text'].notna().sum()} translations from: {output_file}")
            else:
                logger.warning(f"Checkpoint missing required columns ('id','text'): {output_file}")
        except Exception as e:
            logger.warning(f"Could not resume from {output_file}: {e}")

    translated_so_far = int(df_result["text"].notna().sum())
    logger.info(f"Already translated: {translated_so_far}/{len(df_result)}")
    stop_sequences = [
        "\n\n",
        "Here is", "Here's",  # common prefaces
        "Translation:", "Übersetzung:",
        "Explanation:", "Erklärung:",
        "Note:", "Hinweis:",
        "```",
    ]
    for i in tqdm(range(len(df_result)), desc="Translating"):
        if pd.notna(df_result.at[i, "text"]):
            continue  # already translated

        original_text = str(df_result.at[i, "text_source"])
        userMessage = formatTranslationMessage(text=original_text, source_lang=source_lang, target_lang=target_lang)
        try:
            # Generate translation
            translated_text = llm.generate_with_retries( user_message=userMessage, system_message=SYSTEM_MESSAGE, system_role=SYSTEM_ROLE, stop=stop_sequences).strip()

            # Clean quotes if the model adds them
            if translated_text.startswith('"') and translated_text.endswith('"') and len(translated_text) >= 2:
                translated_text = translated_text[1:-1]

            bad_prefixes = ("here is", "translation", "übersetzung", "explanation", "note:")
            if translated_text.lower().lstrip().startswith(bad_prefixes):
                translated_text = translated_text.split("\n", 1)[-1].strip()

            df_result.at[i, "text"] = translated_text

        except Exception as e:
            logger.error(f"Failed at row {i} (id={df_result.at[i, 'id']}): {e}")
            df_result.at[i, "text"] = None

        processed = (i + 1)
        if processed % BATCH_SIZE == 0:
            logger.info(f"Processed {processed} rows -> pausing {BATCH_PAUSE_SECONDS}s to avoid rate limits")
            time.sleep(BATCH_PAUSE_SECONDS)

        if processed % CHECKPOINT_EVERY == 0:
            df_result.to_csv(output_file, index=False)
            logger.info(f"Checkpoint saved: {output_file}")

    # Remove failures
    failed_count = int(df_result["text"].isna().sum())
    if failed_count > 0:
        logger.warning(f"Dropping {failed_count} rows due to translation failure.")
        df_result = df_result.dropna(subset=["text"])

    # SHOW EXAMPLE
    display_samples(df_result, n=5)
    df_result.to_csv(output_file, index=False)

    logger.info("Translation complete.")
    console.print(
        Panel(f"Saved [bold]{len(df_result)}[/bold] translated samples to:\n[underline]{output_file}[/underline]",
              style="bold green"))

if __name__ == "__main__":
    try:
        llm = Client(host=HOST, port=PORT, model_name=MODEL_NAME)
        #llm = Client(base_url=BASE_URL, model_name=MODEL_NAME)  # system role

    except Exception as e:
        logger.error(f"Could not connect to LLM: {e}")
        exit(1)

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        exit(1)

    logger.info(f"Loading data from {INPUT_FILE}...")
    full_df = pd.read_csv(INPUT_FILE)

    en_df = full_df[full_df['language'] == SOURCE_LANG_COL]

    if len(en_df) == 0:
        logger.error(f"No English samples found in {INPUT_FILE}")
        exit(1)

    output_file = Path(OUTPUT_FILE)

    translated_ids = set()
    prev_translated = None

    if output_file.exists():
        prev_translated = pd.read_csv(output_file)
        if "id" in prev_translated.columns and "text" in prev_translated.columns:
            translated_ids = set(prev_translated.dropna(subset=["text"])["id"].astype(str))
        else:
            logger.warning(f"{output_file} exists but missing required columns ('id','text').")

        # Target totals
        n_total = min(SAMPLE_SIZE, len(en_df))
        n_pos_target = min(int(round(n_total * POS_FRAC)), int((en_df["label"] == 1).sum()))
        n_neg_target = n_total - n_pos_target

        # Already translated (within English set)
        already_df = en_df[en_df["id"].isin(translated_ids)]
        already_pos_df = already_df[already_df["label"] == 1]
        already_neg_df = already_df[already_df["label"] == 0]

        already_pos = len(already_pos_df)
        already_neg = len(already_neg_df)

        # If already  more than the target, cap to target while keeping pos/neg ratio
        if (already_pos + already_neg) > n_total:
            take_pos = min(n_pos_target, already_pos)
            take_neg = n_total - take_pos
            take_neg = min(take_neg, already_neg)
            # if still short due to lack of neg, fill with remaining pos (or vice versa)
            remaining = n_total - (take_pos + take_neg)
            if remaining > 0:
                extra_pos = min(remaining, already_pos - take_pos)
                take_pos += extra_pos
                remaining = n_total - (take_pos + take_neg)
            if remaining > 0:
                extra_neg = min(remaining, already_neg - take_neg)
                take_neg += extra_neg

            already_pos_df = already_pos_df.sample(n=take_pos,
                                                   random_state=42) if take_pos > 0 else already_pos_df.iloc[0:0]
            already_neg_df = already_neg_df.sample(n=take_neg,
                                                   random_state=42) if take_neg > 0 else already_neg_df.iloc[0:0]
            already_df = pd.concat([already_pos_df, already_neg_df], ignore_index=True)
            already_pos, already_neg = len(already_pos_df), len(already_neg_df)

        # Remaining needed to reach the target
        need_pos = max(0, n_pos_target - already_pos)
        need_neg = max(0, n_neg_target - already_neg)

        remaining_pool = en_df[~en_df["id"].isin(translated_ids)]

        pos_pool = remaining_pool[remaining_pool["label"] == 1]
        neg_pool = remaining_pool[remaining_pool["label"] == 0]

        need_pos = min(need_pos, len(pos_pool))
        need_neg = min(need_neg, len(neg_pool))

        pos_new = pos_pool.sample(n=need_pos, random_state=42) if need_pos > 0 else pos_pool.iloc[0:0]
        neg_new = neg_pool.sample(n=need_neg, random_state=42) if need_neg > 0 else neg_pool.iloc[0:0]

        df_subset = (
            pd.concat([already_df, pos_new, neg_new], ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

        console.print(
            f"[bold]Using {len(already_df)} already-translated rows + "
            f"{len(pos_new) + len(neg_new)} new rows = {len(df_subset)} total for translation.[/bold]"
        )
    # Run Translation
    translate_dataset(df=df_subset, llm=llm, source_lang=SOURCE_LANG, target_lang=TARGET_LANG)