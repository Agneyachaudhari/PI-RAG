import os
import hashlib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import pickle
from datetime import datetime

# ===== CONFIG =====
DATA_FOLDER = "data"
EMBED_MODEL = "models/embedder"
GGUF_MODEL = "models/llm/model.gguf"
INDEX_FILE = "faiss.index"
META_FILE = "meta.pkl"
LOG_FILE = "chat_log.txt"

MAX_TOKENS = 512
AUTO_CHUNK_MIN = 600
AUTO_CHUNK_MAX = 800
TOP_K_CHUNKS = 3
# Colours cause why not
RESET = "\033[0m"
INFO = "\033[96;1m"    # cyan
GOOD = "\033[92;1m"    # green
WARN = "\033[93;1m"    # yellow
HEAD = "\033[95;1m"    # magenta
ERR = "\033[91;1m"     # red
CHUNK = "\033[97m"     # white
QUEST = "\033[97;1m"   # white bold
ANS = "\033[96;1m"     # cyan bold


def get_text_files():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        raise FileNotFoundError(f"'{DATA_FOLDER}' folder was missing, created it. Add .txt files there and restart.")
    files = sorted([
        os.path.join(DATA_FOLDER, f)
        for f in os.listdir(DATA_FOLDER)
        if f.endswith(".txt")
    ])
    if not files:
        raise FileNotFoundError(f"No .txt files found in '{DATA_FOLDER}/'")
    return files


def write_log(question, answer):
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.now()}]\nQ: {question}\nA: {answer}\n{'-'*40}\n\n")


def file_hash(paths):
    h = hashlib.sha256()
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def auto_chunk(text):
    length = len(text)
    if length < 2000:
        size = AUTO_CHUNK_MIN
    elif length < 6000:
        size = (AUTO_CHUNK_MIN + AUTO_CHUNK_MAX) // 2
    else:
        size = AUTO_CHUNK_MAX
    
    overlap = size // 4  # 25% overlap to avoid cutting sentences
    chunks = []
    
    i = 0
    while i < len(text):
        chunk = text[i:i+size].strip()
        if chunk:
            chunks.append(chunk)
        i += size - overlap
    
    return chunks


def build_or_load_index(embedder, text_files):
    text_hash = file_hash(text_files)
    
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        with open(META_FILE, "rb") as f:
            meta = pickle.load(f)
        
        if meta.get("hash") == text_hash:
            print(f"{GOOD}[OK] Using cached index{RESET}")
            return meta["chunks"], faiss.read_index(INDEX_FILE)
    
    print(f"{WARN}[!] Rebuilding index...{RESET}")
    
    all_chunks = []
    for fpath in text_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors='ignore') as f:
                content = f.read().strip()
                for ch in auto_chunk(content):
                    all_chunks.append((ch, fpath))
        except Exception as e:
            print(f"{ERR}[!] Failed to read {fpath}: {e}{RESET}")
    
    if not all_chunks:
        raise ValueError("No chunks created")
    
    print(f"{INFO}  -> {len(all_chunks)} chunks{RESET}")
    embeddings = embedder.encode([c[0] for c in all_chunks], show_progress_bar=False)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    
    faiss.write_index(index, INDEX_FILE)
    with open(META_FILE, "wb") as f:
        pickle.dump({"hash": text_hash, "chunks": all_chunks}, f)
    
    print(f"{GOOD}[OK] Index saved{RESET}")
    return all_chunks, index


# ===== INITIALIZE =====
try:
    print(f"{INFO}{'='*50}\n{HEAD}   RAG SYSTEM INITIALIZING\n{INFO}{'='*50}{RESET}")

    text_files = get_text_files()
    print(f"{INFO}[*] Found {len(text_files)} file(s) in '{DATA_FOLDER}/': {[os.path.basename(f) for f in text_files]}{RESET}")

    print(f"{INFO}\n[1/3] Loading embedder...{RESET}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"{GOOD}      [OK]{RESET}")
    
    print(f"{INFO}\n[2/3] Loading index...{RESET}")
    chunks, index = build_or_load_index(embedder, text_files)
    print(f"{GOOD}      [OK] {len(chunks)} chunks{RESET}")
    
    print(f"{INFO}\n[3/3] Loading LLM...{RESET}")
    llm = Llama(
        model_path=GGUF_MODEL,
        n_ctx=2500,
        n_threads=4,
        n_batch=128,
        temperature=0.49,
        top_p=0.9,
        repeat_penalty=1.01,
        verbose=False,
        n_gpu_layers=0
    )
    print(f"{GOOD}      [OK]{RESET}")
    
    print(f"{INFO}\n{'='*50}\n{GOOD}   READY{INFO}\n{'='*50}\n{RESET}")
    
except Exception as e:
    print(f"{ERR}\n[ERROR] Init failed: {e}{RESET}")
    exit(1)


def ask(query):
    print(f"\n{HEAD}{'='*50}\n{QUEST}Q: {query}\n{HEAD}{'='*50}{RESET}\n")
    
    qvec = embedder.encode([query], show_progress_bar=False).astype('float32')
    _, idxs = index.search(qvec, TOP_K_CHUNKS)
    
    # Get top K chunks by relevance, don't care about file distribution
    selected = [chunks[i] for i in idxs[0]]
    
    print(f"{HEAD}RELEVANT CHUNKS:{RESET}\n")
    for idx, (chunk, fname) in enumerate(selected):
        preview = chunk[:200].replace("\n", " ")
        print(f"{INFO}[{idx+1}] {fname}{RESET}")
        print(f"{CHUNK}{preview}{'...' if len(chunk) > 200 else ''}{RESET}\n")
    
    context = "\n\n".join([f"From {fname}:\n{chunk}" for chunk, fname in selected])

    prompt = f"""<|start_header_id|>system<|end_header_id|>

Answer the question directly using the information from the paragraphs and ONLY using the paragraphs below.You may modify the paragraphs to fit the expected answer.Do not show your thinking as the user only wants an answer.For example if you get asked for diffrences but only get properties you can give the answer with the properties arranged in a diffrences format and give a direct answer. Be concise and clear.Do not assume or deduce info.That is important. <|eot_id|><|start_header_id|>user<|end_header_id|>

Paragraphs:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    try:
        print(f"{WARN}[...] Generating...{RESET}\n")
        print(f"{ANS}", end='', flush=True)
        
        answer_parts = []
        for output in llm(prompt, max_tokens=MAX_TOKENS, stream=True, stop=["<|eot_id|>", "<|end_of_text|>"]):
            token = output["choices"][0]["text"]
            answer_parts.append(token)
            print(token, end='', flush=True)
        
        print(f"{RESET}\n")
        answer = "".join(answer_parts).strip()
        
        print(f"{HEAD}[{len(answer_parts)} tokens]{RESET}\n")
        write_log(query, answer)
        
    except Exception as e:
        print(f"{ERR}\n[ERROR] {e}{RESET}")


# ===== MAIN =====
print(f"\n{INFO}Type 'exit', 'quit','q' or Ctrl+C to quit{RESET}\n")

while True:
    try:
        q = input(f"{QUEST}ASK: {RESET}").strip()
        if q.lower() in ("exit", "quit", "q"):
            print(f"{WARN}Goodbye!{RESET}")
            llm.close()
            break
        if q:
            ask(q)
    except KeyboardInterrupt:
        print(f"\n{WARN}Exiting...{RESET}")
        break
    except Exception as e:
        print(f"{ERR}[ERROR] {e}{RESET}")
