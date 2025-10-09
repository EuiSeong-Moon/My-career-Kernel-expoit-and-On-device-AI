#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import json
import glob
import numpy as np
import chromadb
from chromadb.config import Settings
from tqdm.auto import tqdm
import uuid
import PyPDF2
from pathlib import Path
from typing import List, Dict, Optional, Iterable, Tuple

from sentence_transformers import SentenceTransformer
import ollama

# ----------------------------
# Config
# ----------------------------
DEFAULT_DB_DIR = "./chroma_db"
DEFAULT_COLLECTION = "security_docs"
DEFAULT_MODEL_PATH = "./st-embeddinggemma-sec"   # your finetuned model dir
DEFAULT_OLLAMA_MODEL = "gemma2"                  # any LLM in your Ollama
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ----------------------------
# IO helpers
# ----------------------------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")

def load_sources(input_glob: str) -> List[Tuple[str, str]]:
    """
    Returns list of (doc_id, text) to index.
    Supports: .pdf only.
    """
    out = []
    for p in glob.glob(input_glob, recursive=True):
        pp = Path(p)
        if pp.is_dir():
            continue
        ext = pp.suffix.lower()
        if ext == ".pdf":
            try:
                with open(pp, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages = []
                    for page in reader.pages:
                        try:
                            pages.append(page.extract_text() or "")
                        except Exception:
                            pages.append("")
                    text = "\n".join(pages)
                out.append((str(pp), text))
            except Exception as e:
                print(f"Failed to read PDF: {pp} ({e})")
        else:
            # ignore other types
            pass
    return out

# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, size - overlap)
    return chunks

# ----------------------------
# Embedding helpers
# ----------------------------
def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    prompt_name: Optional[str] = None,
    normalize: bool = True,
    batch_size: int = 64
) -> np.ndarray:
    """
    If your model supports SBERT v5 prompts, pass prompt_name, e.g.:
      - "Retrieval-query" for queries
      - "Retrieval-document" for documents
    Otherwise leave as None.
    """
    kwargs = dict(normalize_embeddings=normalize, batch_size=batch_size)
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    embs = model.encode(texts, **kwargs)
    return np.array(embs, dtype=np.float32)

# ----------------------------
# Vector DB
# ----------------------------
def get_collection(
    persist_dir: str,
    collection_name: str = DEFAULT_COLLECTION
):
    client = chromadb.Client(
        Settings(
            is_persistent=True,
            persist_directory=persist_dir
        )
    )
    try:
        col = client.get_collection(collection_name)
    except Exception:
        col = client.create_collection(collection_name)
    return col

def add_documents(
    col,
    model: SentenceTransformer,
    docs: List[Tuple[str, str]],
    doc_prompt: Optional[str],
    batch_size: int = 64
):
    ids = []
    metadatas = []
    documents = []
    embeddings = []

    for doc_id, text in tqdm(docs, desc="Chunk & embed"):
        chunks = chunk_text(text)
        if not chunks:
            continue
        # prepare IDs and metas
        cids = [f"{doc_id}::chunk::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
        metas = [{"source": doc_id, "chunk_idx": i} for i in range(len(chunks))]
        # embed
        emb = encode_texts(model, chunks, prompt_name=doc_prompt, normalize=True, batch_size=batch_size)
        # collect
        ids.extend(cids)
        documents.extend(chunks)
        metadatas.extend(metas)
        embeddings.append(emb)

    if not documents:
        print("No documents to add.")
        return

    # flatten embeddings (list of arrays -> one array)
    emb_mat = np.vstack(embeddings)
    # upsert into Chroma
    col.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=emb_mat.tolist()
    )
    print(f"Indexed {len(documents)} chunks from {len(docs)} documents.")

def search(
    col,
    model: SentenceTransformer,
    query: str,
    top_k: int = 4,
    query_prompt: Optional[str] = None
) -> List[Dict]:
    q_emb = encode_texts(model, [query], prompt_name=query_prompt, normalize=True, batch_size=32)[0]
    res = col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "doc": res["documents"][0][i],
            "meta": res["metadatas"][0][i],
            "distance": res["distances"][0][i]
        })
    return hits

# ----------------------------
# Ollama call
# ----------------------------
def ask_ollama(model_name: str, question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    prompt = (
        "You are a precise security engineer. "
        "Answer the question using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    resp = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["message"]["content"].strip()

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx = sub.add_parser("index", help="Index documents to Chroma")
    ap_idx.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap_idx.add_argument("--persist_dir", default=DEFAULT_DB_DIR)
    ap_idx.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap_idx.add_argument("--input_glob", required=True,
                        help="Glob of files to index (e.g., './docs/**/*.md')")
    ap_idx.add_argument("--doc_prompt", default="Retrieval-document",
                        help="SBERT v5 prompt name for documents, or 'none' to disable")
    ap_idx.add_argument("--batch_size", type=int, default=64)

    ap_q = sub.add_parser("query", help="Query with RAG (retrieval + Ollama)")
    ap_q.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    ap_q.add_argument("--persist_dir", default=DEFAULT_DB_DIR)
    ap_q.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap_q.add_argument("--question", required=True)
    ap_q.add_argument("--top_k", type=int, default=4)
    ap_q.add_argument("--query_prompt", default="Retrieval-query",
                      help="SBERT v5 prompt name for queries, or 'none' to disable")
    ap_q.add_argument("--ollama_model", default=DEFAULT_OLLAMA_MODEL)

    args = ap.parse_args()

    # load embedding model
    print(f"Loading embedding model from: {args.model_path}")
    embed_model = SentenceTransformer(args.model_path)

    if args.cmd == "index":
        doc_prompt = None if args.doc_prompt.lower() == "none" else args.doc_prompt
        col = get_collection(args.persist_dir, args.collection)
        docs = load_sources(args.input_glob)
        if not docs:
            print("No documents found for input_glob.")
            return
        add_documents(col, embed_model, docs, doc_prompt=doc_prompt, batch_size=args.batch_size)
        print(f"Persisted DB at: {Path(args.persist_dir).resolve()}")

    elif args.cmd == "query":
        query_prompt = None if args.query_prompt.lower() == "none" else args.query_prompt
        col = get_collection(args.persist_dir, args.collection)
        hits = search(col, embed_model, args.question, top_k=args.top_k, query_prompt=query_prompt)
        if not hits:
            print("No results.")
            return
        contexts = [h["doc"] for h in hits]
        answer = ask_ollama(args.ollama_model, args.question, contexts)
        print("\n=== Retrieved Chunks ===")
        for i, h in enumerate(hits, 1):
            src = h["meta"].get("source", "N/A")
            print(f"[{i}] source={src}  dist={h['distance']:.4f}\n{h['doc']}\n")
        print("=== LLM Answer ===")
        print(answer)

if __name__ == "__main__":
    main()
