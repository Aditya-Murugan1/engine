# train.py
# ─────────────────────────────────────────────────────────────────────────────
# Training launcher for the resume NER model.
# Reads gap_report.json → applies Focal Loss with dynamic class weights.
# Resumes from last checkpoint if available, else starts from dslim/bert-base-NER.
# Saves per-run metrics JSON for UI consumption.
#
# Usage:
#   python train.py                          # standard overnight run
#   python train.py --gap_report path/to/gap_report.json
#   python train.py --fresh                  # force retrain from base model
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import time
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from skill_extractor import ResumeNERDataset, LABELS, LABEL2ID, ID2LABEL, load_resume_texts, load_skill_vocab

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL      = "dslim/bert-base-NER"
TRAIN_FILE      = "data/train.jsonl"
VAL_FILE        = "data/val.jsonl"
OUTPUT_DIR      = "models/skill_extractor"
GAP_REPORT      = "data/gap_report.json"
METRICS_DIR     = "metrics"

MAX_LEN         = 256
BATCH_SIZE      = 16
EPOCHS          = 5
LR              = 2e-5
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.1
FOCAL_GAMMA     = 2.0


from transformers import TrainerCallback

class ThrottleCallback(TrainerCallback):
    """Sleeps 50ms between training steps to keep GPU utilisation ~30%."""
    def on_step_end(self, args, state, control, **kwargs):
        time.sleep(0.05)

# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for token classification.
    Combines per-class weights (from gap_report) with the focal
    (1-p_t)^gamma term that down-weights easy O predictions
    and forces attention onto hard entity tokens.
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = FOCAL_GAMMA):
        super().__init__()
        self.gamma         = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        num_labels  = logits.shape[-1]
        logits_flat = logits.view(-1, num_labels)
        labels_flat = labels.view(-1)

        ce_loss = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device),
            reduction="none",
            ignore_index=-100,
        )(logits_flat, labels_flat)

        probs       = torch.softmax(logits_flat, dim=-1)
        mask        = labels_flat != -100
        safe_labels = labels_flat.clone()
        safe_labels[~mask] = 0

        p_t         = probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
        focal_term  = (1 - p_t) ** self.gamma
        focal_loss  = focal_term * ce_loss

        return focal_loss[mask].mean()

# ── Custom Trainer ────────────────────────────────────────────────────────────

class FocalTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        loss    = self.focal_loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

# ── Class weight loader ───────────────────────────────────────────────────────

def load_class_weights(gap_report_path: str) -> torch.Tensor:
    default = torch.ones(len(LABELS))
    path    = Path(gap_report_path)

    if not path.exists():
        print(f"[WARN] Gap report not found at {gap_report_path} — using uniform weights")
        return default

    try:
        report  = json.loads(path.read_text())
        raw_w   = report["training_hints"]["class_weights"]

        weight_tensor = torch.ones(len(LABELS))
        for i, label in enumerate(LABELS):
            if label == "O":
                weight_tensor[i] = raw_w.get("O", 1.0)
            else:
                entity_type      = label.split("-", 1)[1]
                weight_tensor[i] = raw_w.get(entity_type, 1.0)

        print(f"[INFO] Class weights loaded: {dict(zip(LABELS, weight_tensor.tolist()))}")
        return weight_tensor

    except (KeyError, json.JSONDecodeError) as e:
        print(f"[WARN] Could not parse gap report ({e}) — using uniform weights")
        return default

# ── Checkpoint resolver ───────────────────────────────────────────────────────

def resolve_start_model(output_dir: str, force_fresh: bool) -> tuple:
    if force_fresh:
        print(f"[INFO] --fresh flag set. Starting from {BASE_MODEL}")
        return BASE_MODEL, False

    checkpoints = sorted(Path(output_dir).glob("checkpoint-*")) if Path(output_dir).exists() else []
    if checkpoints:
        latest = str(checkpoints[-1])
        print(f"[INFO] Resuming from checkpoint: {latest}")
        return latest, True

    if (Path(output_dir) / "config.json").exists():
        print(f"[INFO] Resuming from saved model: {output_dir}")
        return output_dir, True

    print(f"[INFO] No checkpoint found. Starting from {BASE_MODEL}")
    return BASE_MODEL, False

# ── Metrics ───────────────────────────────────────────────────────────────────

def build_compute_metrics(run_meta: dict):
    def compute_metrics(p) -> dict:
        preds, label_ids = p
        preds = np.argmax(preds, axis=2)

        true_labels, pred_labels = [], []
        for pred_row, label_row in zip(preds, label_ids):
            tl, pl = [], []
            for p_val, l_val in zip(pred_row, label_row):
                if l_val != -100:
                    tl.append(ID2LABEL[l_val])
                    pl.append(ID2LABEL[p_val])
            true_labels.append(tl)
            pred_labels.append(pl)

        f1        = f1_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels)
        recall    = recall_score(true_labels, pred_labels)
        report    = classification_report(true_labels, pred_labels, output_dict=True)

        run_meta["eval_metrics"] = {
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "per_class": {
                k: {m: round(v, 4) for m, v in v_dict.items()}
                for k, v_dict in report.items()
                if isinstance(v_dict, dict)
            },
        }
        return {"f1": f1, "precision": precision, "recall": recall}
    return compute_metrics

# ── Metrics JSON writer ───────────────────────────────────────────────────────

def save_run_metrics(run_meta: dict, metrics_dir: str):
    """
    Writes metrics/run_TIMESTAMP.json
    UI can glob metrics/run_*.json and sort by run_id to build history.
    """
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(metrics_dir) / f"run_{run_meta['run_id']}.json"
    out_path.write_text(json.dumps(run_meta, indent=2))
    print(f"[INFO] Metrics saved → {out_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Resume NER training launcher")
    parser.add_argument("--gap_report", default=GAP_REPORT)
    parser.add_argument("--fresh",  action="store_true")
    parser.add_argument("--train",  default=TRAIN_FILE)
    parser.add_argument("--val",    default=VAL_FILE)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int,   default=EPOCHS)
    parser.add_argument("--batch",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",     type=float, default=LR)
    args = parser.parse_args()

    run_id   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta = {
        "run_id":          run_id,
        "started_at":      datetime.now().isoformat(),
        "gap_report_path": args.gap_report,
        "training_args": {
            "epochs":      args.epochs,
            "batch_size":  args.batch,
            "lr":          args.lr,
            "max_len":     MAX_LEN,
            "focal_gamma": FOCAL_GAMMA,
        },
    }

    start_model, is_resuming = resolve_start_model(args.output, args.fresh)
    run_meta["resumed_from"] = start_model

    class_weights            = load_class_weights(args.gap_report)
    run_meta["class_weights"] = dict(zip(LABELS, class_weights.tolist()))

    tokenizer = AutoTokenizer.from_pretrained(start_model)
    model     = AutoModelForTokenClassification.from_pretrained(
        start_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # ── Load Person B's data files ────────────────────────────────────────────
    print("[INFO] Loading datasets from Person B's data files...")
    from sklearn.model_selection import train_test_split
    from skill_extractor import load_resume_texts, load_skill_vocab

    texts       = load_resume_texts("data/Resume.csv")
    skill_vocab = load_skill_vocab("data/Skills.xlsx")

    train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
    print(f"[INFO] Split → Train: {len(train_texts)} | Val: {len(val_texts)}")

    train_ds = ResumeNERDataset(train_texts, tokenizer, skill_vocab, MAX_LEN)
    val_ds   = ResumeNERDataset(val_texts,   tokenizer, skill_vocab, MAX_LEN)
    print(f"[INFO] Encoded → Train: {len(train_ds)} | Val: {len(val_ds)}")

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        # batch size set above via load-limiting config
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=100,
        dataloader_num_workers=0,       # 0 = main process only, reduces CPU overhead on Windows
        dataloader_pin_memory=False,    # disable since we now use GPU directly
        per_device_train_batch_size=4,  # small batch = low GPU spike per step
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch still 16, spread across 4 tiny steps
        gradient_checkpointing=True,    # ~40% less VRAM
        fp16=False,                     # disable FP16 — slightly slower but less GPU stress
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        resume_from_checkpoint=start_model if is_resuming else None,
    )

    compute_metrics = build_compute_metrics(run_meta)

    trainer = FocalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[ThrottleCallback()],
    )

    print(f"\n{'='*60}")
    print(f"  Run ID       : {run_id}")
    print(f"  Start model  : {start_model}")
    print(f"  Resuming     : {is_resuming}")
    print(f"  Focal gamma  : {FOCAL_GAMMA}")
    print(f"  Class weights: {dict(zip(LABELS, class_weights.tolist()))}")
    print(f"{'='*60}\n")

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"\n[INFO] Model saved → {args.output}")

    run_meta["finished_at"] = datetime.now().isoformat()
    save_run_metrics(run_meta, METRICS_DIR)

    em = run_meta.get("eval_metrics", {})
    print(f"\n{'='*60}")
    print(f"  F1        : {em.get('f1', 'N/A')}")
    print(f"  Precision : {em.get('precision', 'N/A')}")
    print(f"  Recall    : {em.get('recall', 'N/A')}")
    print(f"\n  Per-class breakdown:")
    for cls, stats in em.get("per_class", {}).items():
        if cls in ("SKILL", "EXPERIENCE", "EDUCATION"):
            print(f"    {cls:<12} P={stats['precision']}  R={stats['recall']}  "
                  f"F1={stats['f1-score']}  support={stats['support']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
