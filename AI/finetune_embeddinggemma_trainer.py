#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune Embedding Gemma using SentenceTransformerTrainer + SentenceTransformerTrainingArguments
(aligns with Google's Embedding Gemma guide)

Supports:
- JSON array of triplets:       [["query","positive","negative"], ...]
- JSON array of objects:        [{"query": "...", "positive": "...", "negative": "..."}, ...]
- JSONL/NDJSON                  (one JSON object per line: query/positive/negative)
- CSV/TSV with headers          query,positive,negative

Loss:
- TripletLoss (default)
- MultipleNegativesRankingLoss (--loss mnr)

Evaluator:
- InformationRetrievalEvaluator when --val_path is provided (computes MAP/NDCG/Recall@k)

Push to Hub:
- pass --hub_repo and set HF_TOKEN
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict

import torch
from huggingface_hub import login
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset

# -----------------------------
# Data loading & normalization
# -----------------------------
def to_triplet_examples(rows: List[Dict[str, str]]) -> List[InputExample]:
    return [InputExample(texts=[r["query"], r["positive"], r["negative"]]) for r in rows]


def to_mnr_pairs(rows: List[Dict[str, str]]) -> List[InputExample]:
    # For MultipleNegativesRankingLoss: (query, positive) pairs; negatives are in-batch.
    return [InputExample(texts=[r["query"], r["positive"]]) for r in rows]


def make_ir_evaluator(rows: List[Dict[str, str]], name: str = "dev") -> evaluation.InformationRetrievalEvaluator:
    queries, corpus, relevant_docs = {}, {}, {}
    doc_id = 0

    def add_doc(txt: str) -> str:
        nonlocal doc_id
        did = f"d{doc_id}"
        corpus[did] = txt
        doc_id += 1
        return did

    for i, r in enumerate(rows):
        qid = f"q{i}"
        queries[qid] = r["query"]
        pos_id = add_doc(r["positive"])
        _ = add_doc(r["negative"])
        relevant_docs[qid] = {pos_id: 1}

    return evaluation.InformationRetrievalEvaluator(queries, corpus, relevant_docs, name=name)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", required=True, help="Training data path")
    ap.add_argument("--val_path", default=None, help="Optional dev/validation data path")
    ap.add_argument("--base_model", default="google/embeddinggemma-300m",
                    help="ST-compatible Embedding Gemma checkpoint")
    ap.add_argument("--output_dir", default="./st-embeddinggemma-finetuned")
    ap.add_argument("--loss", choices=["triplet", "mnr"], default="triplet")
    ap.add_argument("--max_seq_length", type=int, default=256)

    # TrainingArguments (trainer-style)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--hub_repo", default=None, help="If set, push to this HF repo (requires HF_TOKEN)")
    args = ap.parse_args()

    # 1) Load datasets
    with open(args.train_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Convert to dicts with keys: anchor, positive, negative
    data_as_dicts = []
    for row in dataset:
        if isinstance(row, dict):
            # If already a dict with keys
            data_as_dicts.append({
                "anchor": row.get("query", row.get("anchor")),
                "positive": row.get("positive"),
                "negative": row.get("negative")
            })
        elif isinstance(row, list) and len(row) == 3:
            # If a list: [query, positive, negative]
            data_as_dicts.append({
                "anchor": row[0],
                "positive": row[1],
                "negative": row[2]
            })
        else:
            raise ValueError("Each row must be a dict or a list of length 3.")
    train_dataset = Dataset.from_list(data_as_dicts)
    print(train_dataset)

    # 2) Build InputExamples
    if args.loss == "triplet":
        loss_obj = losses.TripletLoss
    else:
        loss_obj = losses.MultipleNegativesRankingLoss

    # 3) Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.base_model, device=device)
    model.max_seq_length = args.max_seq_length
    # 4) Loss instance bound to model
    loss_fn = loss_obj(model)
    # 5) Training arguments (trainer-style, per Google guide & SBERT v3)
    #    See: sbert TrainingArguments & Trainer docs
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        # per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        tf32=args.tf32,
        # ⬇️ 여기서 evaluation_strategy 제거
        # evaluation_strategy="steps" if evaluator else "no",
        # 대신 eval_steps만 유지 (트레이너가 이 값을 사용)
        # eval_steps=args.eval_steps if evaluator else None,
        # save_strategy="steps",
        # save_steps=args.save_steps,
        # save_total_limit=2,
        logging_steps=train_dataset.num_rows,
        seed=args.seed,
        report_to=[],                  # ["tensorboard"] or ["wandb"]
        # push_to_hub=bool(args.hub_repo),
        # hub_model_id=args.hub_repo if args.hub_repo else None,
       # dataloader_drop_last=True,
    )


    # 6) Trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_ds,
        loss=loss_fn,
        # evaluator=evaluator,  # computes IR metrics on --val_path
    )

    # 7) Train
    trainer.train()
    trainer.save_model(args.output_dir)

    # Normalize embeddings by default for cosine similarity downstream
    cfg = Path(args.output_dir) / "config_sentence_transformers.json"
    if cfg.exists():
        j = json.loads(cfg.read_text(encoding="utf-8"))
        j["normalize_embeddings"] = True
        cfg.write_text(json.dumps(j, indent=2), encoding="utf-8")

    # 8) Optional push to Hub
    if args.hub_repo:
        trainer.push_to_hub()

    print(f"Done. Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
