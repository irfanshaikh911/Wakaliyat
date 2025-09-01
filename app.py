# ============================== #
# Legal RAG with Map-Reduce Summarization + Persistence (CPU FAISS)
# ============================== #
from datasets import load_dataset
import pandas as pd
import re, string, json, os
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------- Config --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME   = "google/flan-t5-base"   # text2text; works for both summarization and answering
TOP_K_RETRIEVE   = 12
TOP_M_COMPRESS   = 5
DATASET_NAME     = "mratanusarkar/Indian-Laws"

ARTIFACT_DIR     = "./artifacts_legal_rag"
SENTENCES_PATH   = os.path.join(ARTIFACT_DIR, "sentences.json")
FAISS_INDEX_PATH = os.path.join(ARTIFACT_DIR, "faiss.index")

os.makedirs(ARTIFACT_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# -------------------- Preprocessing --------------------
def preprocess_legal_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)   # page numbers
    text = re.sub(r"_+", "", text)                # underscores
    text = re.sub(r"-{2,}", "", text)             # long dashes
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    return text.strip()

# -------------------- Load / Build Corpus --------------------
def load_or_build_sentences():
    if os.path.exists(SENTENCES_PATH):
        with open(SENTENCES_PATH, "r", encoding="utf-8") as f:
            sents = json.load(f)
        print(f"Loaded {len(sents)} preprocessed sentences from disk.")
        return sents

    ds = load_dataset(DATASET_NAME)
    df = pd.DataFrame(ds["train"])
    texts = []
    for col in ["act_title", "section", "law"]:
        if col in df.columns:
            texts.extend(df[col].dropna().astype(str).tolist())

    sents = list({preprocess_legal_text(t) for t in texts if isinstance(t, str) and t.strip()})
    with open(SENTENCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sents, f, ensure_ascii=False)
    print(f"Built and saved {len(sents)} preprocessed sentences.")
    return sents

sentences = load_or_build_sentences()

# -------------------- Embeddings + FAISS (CPU only) --------------------
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

def build_faiss_index(sents):
    embs = embedder.encode(sents, convert_to_numpy=True, show_progress_bar=True).astype("float32")
    dim = embs.shape[1]
    cpu_index = faiss.IndexFlatL2(dim)
    cpu_index.add(embs)
    faiss.write_index(cpu_index, FAISS_INDEX_PATH)
    print(f"Built FAISS (CPU) with {cpu_index.ntotal} vectors. Saved to disk.")
    return cpu_index

def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    cpu_index = faiss.read_index(FAISS_INDEX_PATH)
    print(f"Loaded FAISS index (CPU) with {cpu_index.ntotal} vectors.")
    return cpu_index

index = load_faiss_index() or build_faiss_index(sentences)

# -------------------- HF Pipeline (summarize + answer) --------------------
t2t = pipeline(
    "text2text-generation",
    model=LLM_MODEL_NAME,
    device=0 if DEVICE == "cuda" else -1
)

# -------------------- Retrieval --------------------
def retrieve(query: str, k: int = TOP_K_RETRIEVE):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, k)
    hits = [(float(distances[0][i]), int(indices[0][i])) for i in range(len(indices[0]))]
    hits.sort(key=lambda x: x[0])  # smaller distance = closer
    return hits

# -------------------- Map-Reduce Compression --------------------
def compress_chunks(chunks, m: int = TOP_M_COMPRESS):
    chunks = chunks[:m]
    mini_summaries = []
    for ch in chunks:
        prompt = f"Summarize this legal text briefly and clearly:\n{ch}\nSummary:"
        out = t2t(prompt, max_new_tokens=80, clean_up_tokenization_spaces=True)[0]["generated_text"]
        mini_summaries.append(out.strip())

    combined = "\n".join(mini_summaries)
    reduce_prompt = f"Combine the following brief legal summaries into a concise, coherent context (3-5 lines):\n{combined}\nConcise context:"
    reduced = t2t(reduce_prompt, max_new_tokens=120, clean_up_tokenization_spaces=True)[0]["generated_text"].strip()
    return reduced, mini_summaries

# -------------------- Ask function --------------------
def ask_legal_question(query: str, k: int = TOP_K_RETRIEVE, m: int = TOP_M_COMPRESS):
    hits = retrieve(query, k=k)
    retrieved_texts = [sentences[idx] for _, idx in hits]

    compressed_context, mini_summaries = compress_chunks(retrieved_texts, m=m)

    answer_prompt = f"""
You are a legal assistant.
Use ONLY the provided compressed legal context to answer the user's query.
If the answer cannot be found, reply exactly: "Not found in the provided laws".

Query: {query}
Compressed legal context:
{compressed_context}

Answer:
"""
    final = t2t(answer_prompt, max_new_tokens=220, clean_up_tokenization_spaces=True)[0]["generated_text"].strip()

    return {
        "answer": final,
        "mini_summaries": mini_summaries,
        "compressed_context": compressed_context,
        "raw_matches": retrieved_texts[:m]
    }

# -------------------- Example --------------------
if __name__ == "__main__":
    q = "Penalty for sale"
    result = ask_legal_question(q)
    print("\n=== QUERY ===")
    print(q)
    print("\n=== ANSWER ===")
    print(result["answer"])
    print("\n=== COMPRESSED CONTEXT ===")
    print(result["compressed_context"])
    print("\n=== TOP MINI-SUMMARIES (MAP) ===")
    for i, s in enumerate(result["mini_summaries"], 1):
        print(f"{i}. {s}")
