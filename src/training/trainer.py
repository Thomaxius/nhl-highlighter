"""
training/trainer.py

Fine-tunes a VideoMAE model on your labelled NHL segments using the
HuggingFace Trainer API with wandb experiment tracking.

Usage
-----
    python -m src.training.trainer \
        --data_dir data/labeled \
        --output_dir models/checkpoints/videomae_nhl \
        --epochs 10 \
        --batch_size 4
"""

from pathlib import Path
import logging
import argparse
import torch
from transformers import (
    VideoMAEForVideoClassification,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import DataLoader
import numpy as np

from src.training.dataset import NHLSegmentDataset

logger = logging.getLogger(__name__)


def compute_class_weights(dataset: "NHLSegmentDataset") -> torch.Tensor:
    """Compute inverse-frequency class weights from a dataset's sample list.

    Classes with 0 samples in this split are assigned weight 0.0 so they
    don't pollute the loss (they can't appear in a batch anyway).
    """
    from collections import Counter
    counts = Counter(label_idx for _, label_idx in dataset.samples)
    num_classes = len(dataset.labels)
    total = len(dataset.samples)
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 0)
        if count == 0:
            weights.append(0.0)  # class absent from this split — ignore
        else:
            weights.append(total / (num_classes * count))
    weight_tensor = torch.tensor(weights, dtype=torch.float)
    for label, i in dataset.label2idx.items():
        logger.info("  Class weight  %-20s  n=%-4d  w=%.3f", label, counts.get(i, 0), weights[i])
    return weight_tensor


class WeightedTrainer(Trainer):
    """HuggingFace Trainer with inverse-frequency class weighting."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1}


def collate_fn(batch):
    """Stack variable-length video batches for the HuggingFace Trainer."""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    # VideoMAE expects [B, T, C, H, W] — permute from [B, T, C, H, W] as is
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    base_model: str = "MCG-NJU/videomae-base",
    num_frames: int = 16,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 5e-5,
    use_wandb: bool = True,
) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # ── Discover labels ──────────────────────────────────────────────────────
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    logger.info("Labels: %s", labels)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_dataset = NHLSegmentDataset(data_dir, labels=labels, num_frames=num_frames, split="train")
    val_dataset = NHLSegmentDataset(data_dir, labels=labels, num_frames=num_frames, split="val")

    # ── Model ────────────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(base_model)
    model = VideoMAEForVideoClassification.from_pretrained(
        base_model,
        num_labels=len(labels),
        label2id={l: i for i, l in enumerate(labels)},
        id2label={i: l for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True,
    )

    # ── Training Arguments ───────────────────────────────────────────────────
    report_to = "wandb" if use_wandb else "none"

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=report_to,
        run_name="videomae-nhl-highlighter",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,        # 2 workers keeps RAM pressure low on 16 GB
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
    )

    # ── Class weights ────────────────────────────────────────────────────────
    logger.info("Computing class weights from training set...")
    class_weights = compute_class_weights(train_dataset)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Model saved to %s", output_dir)


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Fine-tune VideoMAE on NHL segments")
    p.add_argument("--data_dir", default="data/labeled")
    p.add_argument("--output_dir", default="models/checkpoints/videomae_nhl")
    p.add_argument("--base_model", default="MCG-NJU/videomae-base")
    p.add_argument("--num_frames", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_frames=args.num_frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=not args.no_wandb,
    )
