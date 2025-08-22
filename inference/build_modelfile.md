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
