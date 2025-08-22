# RL-MemTool-Agent
RL-style “reason–then-tool” agent with unified memory management—plus how to use the resulting policy locally with Ollama.


## 2) Directory Structure

```javascript
RL-MemTool-Agent/
├─ configs/
│  ├─ train.yaml
│  └─ eval.yaml
├─ data/
│  ├─ download_hotpot.py
│  └─ prepare_hotpot.py
├─ agent/
│  ├─ policy.py
│  ├─ memory_manager.py
│  ├─ tools.py
│  ├─ orchestration.py
│  └─ prompts.py
├─ rl/
│  ├─ env.py
│  ├─ rewards.py
│  └─ train_ppo.py
├─ eval/
│  ├─ metrics.py
│  └─ evaluate.py
├─ inference/
│  ├─ run_local_agent.py
│  ├─ ollama_loop.py
│  ├─ build_modelfile.md
│  └─ serve_api.py
├─ baselines/
│  ├─ vanilla_llm.py
│  └─ rag_baseline.py
├─ scripts/
│  ├─ train.sh
│  ├─ eval.sh
│  └─ export_merge_to_gguf.sh
├─ requirements.txt
└─ README.md

```


## 3) Environment & install
```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

```

## 10) Training / validation / test scripts

- scripts/train.sh
```bash
#!/usr/bin/env bash
set -e
source .venv/bin/activate

python data/download_hotpot.py
python data/prepare_hotpot.py
accelerate launch rl/train_ppo.py --config configs/train.yaml

```

- scripts/eval.sh
```bash
#!/usr/bin/env bash
set -e
source .venv/bin/activate
python eval/evaluate.py --cfg configs/train.yaml
```


### 12.2 Create an Ollama **Modelfile**

**inference/build\_modelfile.md**

```md
Create `Modelfile` like:

FROM ./outputs/gguf/memtool-tinyllama.Q4_K_M.gguf
TEMPLATE """
<|system|>
You are MemToolAgent running under Ollama. Follow the format:
- THINK notes
- optional TOOL JSON: {"action": "...", "args": {...}}
- FINAL_ANSWER: <text>
<|user|>
{{ .Prompt }}
<|assistant|>
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

Then:

```bash
ollama create memtool-agent -f Modelfile
ollama run memtool-agent "Who wrote the novel Dune? If unsure, consider using a tool."
```

> **Note:** Ollama itself does **not** execute tools. Your **orchestration loop** remains in Python, calling Ollama over HTTP and deciding when to run tools. Use `inference/ollama_loop.py`.



---

# 13) Performance measurements & comparison

**Indices to report:**

* **EM (Exact Match)**
* **F1**
* **Tool Use Rate** (% of queries where at least one tool called)
* **Avg Tools / Query**
* **Avg Steps / Query**
* **Latency** (ms) – measure end-to-end per query
* **Token Budget Breaches** (# of times memory had to evict)
* **Hallucination Guard** (optional heuristic: % answers containing “according to X” when X not observed)

**Compare**:

1. **Vanilla LLM** (no tools, no memory)
2. **RAG Baseline** (BM25 over contexts, then generate)
3. **Your RL Agent** (learned tool choice + memory mgmt)

Use `eval/evaluate.py` for (3) and similar scripts for (1) and (2). Combine outputs into a single CSV and print a table of means. You can add importance-weighted deltas: `ΔEM`, `ΔF1`, and **cost metrics** like tokens generated and HTTP calls.

---

# 14) How to run end-to-end

```bash
# 0) Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Data
python data/download_hotpot.py
python data/prepare_hotpot.py

# 2) Train (PPO+QLoRA)
bash scripts/train.sh

# 3) Evaluate and compare
bash scripts/eval.sh
# (add baselines scripts similarly; then merge metrics)

# 4) Merge LoRA and convert to GGUF for Ollama
bash scripts/export_merge_to_gguf.sh
# run the printed llama.cpp conversion command

# 5) Build and run in Ollama
# in inference/build_modelfile.md follow steps:
ollama create memtool-agent -f Modelfile
python inference/ollama_loop.py   # orchestration that uses Ollama HTTP
```

---

