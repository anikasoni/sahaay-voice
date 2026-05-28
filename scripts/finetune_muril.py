"""Fine-tune MuRIL (or any HF classifier) on the Sahaay intents.

Usage:
    # 1. Build / extend `data/intent_train.jsonl` — one JSON per line:
    #    {"text": "main theek hoon", "intent": "safety_confirmation"}
    # 2. Run:
    #    python scripts/finetune_muril.py --epochs 3 --batch 16
    # 3. Flip on the fine-tuned model:
    #    export SAHAAY_FINETUNED=1
    #    streamlit run app/main.py

A starter seed dataset (~6 examples per intent) is built from
config/intents.yaml automatically if no jsonl is found — enough to smoke-test
the pipeline but NOT enough to outperform zero-shot. Extend it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import intents  # noqa: E402


def build_seed_dataset(out_path: Path) -> None:
    rows = []
    cfg = intents()["intents"]
    for intent_name, spec in cfg.items():
        for lang_examples in spec.get("examples", {}).values():
            for ex in lang_examples:
                rows.append({"text": ex, "intent": intent_name})
    random.shuffle(rows)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote seed dataset → {out_path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "intent_train.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "muril_intent")
    ap.add_argument("--base", type=str, default="google/muril-base-cased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    if not args.data.exists():
        args.data.parent.mkdir(parents=True, exist_ok=True)
        build_seed_dataset(args.data)
        print("⚠️  Seed data is tiny. Add more examples before training seriously.")

    # Imports deferred so this script's --help is fast.
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    rows = [json.loads(l) for l in args.data.read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = sorted({r["intent"] for r in rows})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    for r in rows:
        r["label"] = label2id[r["intent"]]

    random.seed(42)
    random.shuffle(rows)
    split = int(0.85 * len(rows))
    train, val = rows[:split], rows[split:]
    ds = {"train": Dataset.from_list(train), "validation": Dataset.from_list(val)}

    tok = AutoTokenizer.from_pretrained(args.base)
    def _tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=128)
    ds = {k: v.map(_tokenize, batched=True) for k, v in ds.items()}
    collator = DataCollatorWithPadding(tok)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(labels), id2label=id2label, label2id=label2id,
    )

    def metrics(p):
        from sklearn.metrics import accuracy_score, f1_score
        preds = np.argmax(p.predictions, axis=-1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1_macro": f1_score(p.label_ids, preds, average="macro"),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=metrics,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"saved fine-tuned model → {args.out}")
    print("set SAHAAY_FINETUNED=1 to use it.")


if __name__ == "__main__":
    main()
