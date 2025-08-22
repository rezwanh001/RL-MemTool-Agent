"""
Agent loop that uses Ollama's HTTP API instead of HF transformers locally.
"""
import requests, re, json, time
from agent.tools import WebSearchTool, RetrievalTool
from agent.memory_manager import MemoryManager
from agent.prompts import SYSTEM_PROMPT

OLLAMA_URL = "http://localhost:11434/api/generate"
FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.I)

def ollama_generate(model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"]

def run_with_ollama(model_name, question, candidate_contexts=None):
    mem = MemoryManager(budget_tokens=1800, tokenizer=None, priority_keys=["evidence","plan"])
    tools = {"web_search": WebSearchTool()}
    if candidate_contexts:
        tools["retrieve"] = RetrievalTool(candidate_contexts)

    for _ in range(5):
        prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|memory|>\n{mem.snapshot()}\n<|user|>\n{question}\n<|assistant|>\n"
        reply = ollama_generate(model_name, prompt)
        try:
            blob = json.loads(re.search(r"\{.*\}", reply, flags=re.S).group(0))
            tool = tools.get(blob.get("action"))
            if tool:
                res = tool.call(**blob.get("args", {}))
                mem.add("evidence", res)
                continue
        except Exception:
            pass
        mem.add("note", reply)
        m = FINAL_RE.search(reply)
        if m:
            return m.group(1).strip()
    return "No confident answer."

if __name__ == "__main__":
    print(run_with_ollama("memtool-agent", "Who won the 2019 NBA championship?"))
