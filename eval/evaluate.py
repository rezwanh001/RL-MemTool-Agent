"""
Runs the trained agent on the validation set and writes metrics + a CSV.
"""
import csv, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent.policy import LMPolicy
from agent.orchestration import Orchestrator
from agent.tools import RetrievalTool, WebSearchTool
from eval.metrics import aggregate
import yaml

def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def main(cfg_path="configs/train.yaml", ckpt_dir="outputs/tinyllama-hotpot-ppo"):
    cfg = yaml.safe_load(open(cfg_path))
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForCausalLM.from_pretrained(ckpt_dir, device_map="auto")
    policy = LMPolicy(cfg["model_name"])
    policy.model, policy.tok = model, tok

    val = load_records("data/prepared/valid.jsonl")
    rows, results = [], []

    for ex in val[:200]:
        tools = {}
        if cfg["tools"]["enable_retrieval"]:
            tools["retrieve"] = RetrievalTool(ex["candidate_contexts"])
        if cfg["tools"]["enable_web_search"]:
            tools["web_search"] = WebSearchTool()

        orch = Orchestrator(policy, tok, tools, cfg["memory"])
        pred = orch.run(ex["question"], max_steps=5)
        steps = len(orch.memory.notes)
        tool_calls = sum(1 for n in orch.memory.notes if n.kind == "evidence")
        results.append({"pred": pred, "gold": ex["gold_answer"], "steps": steps, "tool_calls": tool_calls})
        rows.append([ex["eval_id"], ex["question"], pred, ex["gold_answer"], steps, tool_calls])

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/val_results.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["id","question","pred","gold","steps","tool_calls"])
        w.writerows(rows)

    print(aggregate(results))

if __name__ == "__main__":
    main()
