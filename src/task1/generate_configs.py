import json
from pathlib import Path

import logging
from rich.console import Console
from src.utils import get_project_root

# Setup Logger
logger = logging.getLogger("Config")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = Console()

project_root = get_project_root()

def save_config(name, config):
    path = Path(project_root / f"configs/{name}.json")
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Created {path}")

base_config = {
    "project_name": "SMM4H_Task1",
    "model_name": "xlm-roberta-base",
    "num_labels": 2,
    "id2label": {0: "NoADE", 1: "ADE"},
    "label2id": {"NoADE": 0, "ADE": 1},
    "tokenization": {
        "max_length": 256,
        "truncation": True,
        "padding": "max_length"
    },
    "training_args": {
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "learning_rate": 1e-5,
        "warmup_steps": 0.06,
        "lr_scheduler_type": "linear",
        "max_grad_norm": 1.0,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "num_train_epochs": 20,
        "weight_decay": 0.05,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_steps": 100,
        "seed": 42,
        "report_to": "none",
        "use_cpu": True,
        "fp16": False,
        "bf16": False
    },
    "imbalance_handling": {"use_class_weights": True},
    "early_stopping": {"enabled": True, "patience": 5}
}

# Monolingual English
cfg_en = base_config.copy()
cfg_en["output_dir"] = "output/task1/mono_en"
cfg_en["dataset"] = {
    "data_dir": "data/SMM4H_2026",
    "train_filename": "train.csv",
    "val_filename": "val.csv",
    "test_filename": "test.csv",
    "text_column": "text", "label_column": "label",
    "train_lang": ["en"], # ONLY ENGLISH
    "eval_lang": ["en", "de"],
}
save_config("task1-mono-en", cfg_en)


# Monolingual Russian
cfg_ru = base_config.copy()
cfg_ru["output_dir"] = "output/task1/mono_ru"
cfg_ru["dataset"] = {
    "data_dir": "data/SMM4H_2026",
    "train_filename": "train.csv",
    "val_filename": "val.csv",
    "test_filename": "test.csv",
    "text_column": "text", "label_column": "label",
    "train_lang": ["ru"], # ONLY RUSSIAN
    "eval_lang": ["ru"],
}
save_config("task1-mono-ru", cfg_ru)

# Multilingual (En + Ru)
cfg_multi = base_config.copy()
cfg_multi["output_dir"] = "output/task1/multi_en_ru"
cfg_multi["dataset"] = {
    "data_dir": "data/SMM4H_2026",
    "train_filename": "train.csv",
    "val_filename": "val.csv",
    "test_filename": "test.csv",
    "text_column": "text", "label_column": "label",
    "train_lang": ["en", "ru"], # COMBINED
    "eval_lang": ["en", "ru", "de"],
}
save_config("task1-multi-en-ru", cfg_multi)


# Translate-Train (German)
cfg_trans = base_config.copy()
cfg_trans["output_dir"] = "output/task1/mono_trans_de"
cfg_trans["dataset"] = {
    "data_dir": "data/SMM4H_2026",
    "train_filename": "train_de_translated.csv",
    "val_filename": "val.csv",
    "test_filename": "test.csv",
    "text_column": "text", "label_column": "label",
    "train_lang" : None, # No filter
    "multi_trans" : False, # Only text col for de
    "eval_lang": ["de"],
}
save_config("task1-mono-trans-de", cfg_trans)

# Multilingual Translate-Train (German) & English
cfg_multi_trans = base_config.copy()
cfg_multi_trans["output_dir"] = "output/task1/multi_en_trans_de"
cfg_multi_trans["dataset"] = {
    "data_dir": "data/SMM4H_2026",
    "train_filename": "train_de_translated.csv",
    "val_filename": "val.csv",
    "test_filename": "test.csv",
    "text_column": "text", "label_column": "label",
    "train_lang": None, # No filter
    "multi_trans" : True, # text col for de, text_source col for en
    "eval_lang": ["en", "de"],
}
save_config("task1-multi-en-trans-de", cfg_multi_trans)