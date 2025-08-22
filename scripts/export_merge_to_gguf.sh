#!/usr/bin/env bash
set -e
CKPT_DIR="outputs/tinyllama-hotpot-ppo"
MERGED_DIR="outputs/merged_full"
GGUF_DIR="outputs/gguf"

# 1) Merge LoRA into base
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os, shutil

base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter = "outputs/tinyllama-hotpot-ppo"

tok = AutoTokenizer.from_pretrained(base, use_fast=True)
base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter)
merged = model.merge_and_unload()

os.makedirs("outputs/merged_full", exist_ok=True)
merged.save_pretrained("outputs/merged_full")
tok.save_pretrained("outputs/merged_full")
print("Merged weights saved to outputs/merged_full")
PY

# 2) Convert to GGUF with llama.cpp (requires you to have llama.cpp checked out)
# Example:
# python /path/to/llama.cpp/convert-hf-to-gguf.py outputs/merged_full --outfile outputs/gguf/memtool-tinyllama.Q4_K_M.gguf --quantize q4_k_m

mkdir -p "$GGUF_DIR"
echo "Now run llama.cpp conversion, e.g.:"
echo "python llama.cpp/convert-hf-to-gguf.py outputs/merged_full --outfile outputs/gguf/memtool-tinyllama.Q4_K_M.gguf --quantize q4_k_m"
