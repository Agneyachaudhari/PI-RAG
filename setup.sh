#!/bin/bash

RESET="\033[0m"
INFO="\033[96;1m"
GOOD="\033[92;1m"
WARN="\033[93;1m"
ERR="\033[91;1m"
HEAD="\033[95;1m"

VENV_NAME="rag-env"
EMBEDDER_NAME="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDER_SAVE="models/embedder"
LLM_DIR="models/llm"
LLM_FILE="$LLM_DIR/model.gguf"

LLM_URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0_8_8.gguf"

set -e  
echo -e "${INFO}================================================${RESET}"
echo -e "${HEAD}         RAG ENVIRONMENT SETUP                  ${RESET}"
echo -e "${INFO}================================================${RESET}"

# ===== STEP 1: System dependencies =====
echo -e "\n${INFO}[1/5] Installing system dependencies...${RESET}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    git \
    wget \
    libopenblas-dev
echo -e "${GOOD}      [OK]${RESET}"

# ===== STEP 2: Create venv =====
echo -e "\n${INFO}[2/5] Creating virtual environment '${VENV_NAME}'...${RESET}"
if [ -d "$VENV_NAME" ]; then
    echo -e "${WARN}      [!] venv already exists, skipping creation${RESET}"
else
    python3 -m venv "$VENV_NAME"
    echo -e "${GOOD}      [OK]${RESET}"
fi

# Activate venv
source "$VENV_NAME/bin/activate"
echo -e "${GOOD}      [OK] Activated${RESET}"

# ===== STEP 3: Install Python libraries =====
echo -e "\n${INFO}[3/5] Installing Python libraries...${RESET}"
pip install --upgrade pip --quiet

pip install \
    numpy \
    faiss-cpu \
    sentence-transformers

# llama-cpp-python CPU only build (no CUDA/Metal)
echo -e "${INFO}      Installing llama-cpp-python (CPU only, this takes a while on Pi)...${RESET}"
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
pip install llama-cpp-python --no-binary llama-cpp-python

echo -e "${GOOD}      [OK]${RESET}"

# ===== STEP 4: Download and cache embedder =====
echo -e "\n${INFO}[4/5] Downloading embedder (${EMBEDDER_NAME})...${RESET}"

if [ -d "$EMBEDDER_SAVE" ] && [ "$(ls -A $EMBEDDER_SAVE)" ]; then
    echo -e "${WARN}      [!] Embedder already exists at ${EMBEDDER_SAVE}, skipping${RESET}"
else
    mkdir -p "$EMBEDDER_SAVE"
    python3 - <<EOF
from sentence_transformers import SentenceTransformer
import sys

print("  Downloading model...")
model = SentenceTransformer("${EMBEDDER_NAME}")
model.save("${EMBEDDER_SAVE}")
print("  Saved to ${EMBEDDER_SAVE}")
EOF
    echo -e "${GOOD}      [OK]${RESET}"
fi

# ===== STEP 5: Download LLM =====
echo -e "\n${INFO}[5/5] Downloading LLM model...${RESET}"

if [ -f "$LLM_FILE" ]; then
    echo -e "${WARN}      [!] model.gguf already exists, skipping${RESET}"
else
    if [ "$LLM_URL" = "PASTE_YOUR_GGUF_DOWNLOAD_LINK_HERE" ]; then
        echo -e "${WARN}      [!] No LLM URL set. Edit LLM_URL in this script and re-run.${RESET}"
        echo -e "${WARN}          Skipping LLM download for now.${RESET}"
    else
        mkdir -p "$LLM_DIR"
        echo -e "${INFO}      Downloading to ${LLM_FILE}...${RESET}"
        wget -O "$LLM_FILE" "$LLM_URL" --progress=bar:force
        echo -e "${GOOD}      [OK]${RESET}"
    fi
fi

# ===== DONE =====
echo -e "\n${INFO}================================================${RESET}"
echo -e "${GOOD}   SETUP COMPLETE${RESET}"
echo -e "${INFO}================================================${RESET}"
echo -e "\n${INFO}To activate the environment later:${RESET}"
echo -e "    source ${VENV_NAME}/bin/activate"
echo -e "\n${INFO}To run the RAG system:${RESET}"
echo -e "    source ${VENV_NAME}/bin/activate && python3 script.py\n"
