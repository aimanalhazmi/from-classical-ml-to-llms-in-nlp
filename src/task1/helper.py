import time
import numpy as np
from transformers import TrainerCallback
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging
from rich.console import Console
from rich.panel import Panel

# Setup
console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TimingCallback(TrainerCallback):
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.train_start = None
        self.epoch_start = None
        self.epoch_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.perf_counter()
        self.epoch_times = []
        self.logger.info("Training started.")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()
        self.logger.info(f"Epoch {state.epoch + 1} started.")

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start is None:
            return
        dur = time.perf_counter() - self.epoch_start
        self.epoch_times.append({"epoch": state.epoch, "seconds": dur})
        self.logger.info(f"Epoch {state.epoch} finished in {dur:.2f}s.")

    def on_train_end(self, args, state, control, **kwargs):
        if self.train_start is None:
            return
        total = time.perf_counter() - self.train_start
        self.logger.info(f"Training finished in {total:.2f}s.")


def print_threshold_sanity(scores, labels, threshold, title="Score / Threshold Sanity Check"):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(int)

    pred_mask = scores >= threshold
    pred_rate = float(pred_mask.mean())
    true_rate = float(labels.mean())

    pred_pos = int(pred_mask.sum())
    true_pos = int(labels.sum())
    n = int(len(labels))

    over_factor = pred_rate / (true_rate + 1e-12)

    console.print(Panel(
        "\n".join([
            f"Threshold: {threshold:.3f}",
            f"Scores: min={scores.min():.4f} | mean={scores.mean():.4f} | max={scores.max():.4f}",
            f"Predicted positives: {pred_pos}/{n} ({pred_rate*100:.2f}%)",
            f"True positives:      {true_pos}/{n} ({true_rate*100:.2f}%)",
            f"Over prediction:     {over_factor:.2f}x",
        ]),
        title=title
    ))

def find_best_threshold(y_true, pos_scores):
    best = {"t": 0.5, "f1": -1, "p": 0, "r": 0}
    for t in np.linspace(0.01, 0.99, 99):
        y_pred = (pos_scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {"t": float(t), "f1": float(f1), "precision": float(p), "recall": float(r)}
    return best



def compute_metrics_classic(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.argmax(preds, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
