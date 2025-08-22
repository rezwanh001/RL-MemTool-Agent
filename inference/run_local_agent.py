"""
Loads your checkpoint and runs interactive Q&A with tool orchestration.
"""
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from agent.policy import LMPolicy
from agent.tools import RetrievalTool, WebSearchTool
from agent.orchestration import Orchestrator

if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/train.yaml"))
    tok = AutoTokenizer.from_pretrained("outputs/tinyllama-hotpot-ppo")
    model = AutoModelForCausalLM.from_pretrained("outputs/tinyllama-hotpot-ppo", device_map="auto")
    policy = LMPolicy(cfg["model_name"])
    policy.model, policy.tok = model, tok

    print("Ready. Ask a question (CTRL+C to quit).")
    while True:
        q = input(">> ")
        tools = {"web_search": WebSearchTool()}
        orch = Orchestrator(policy, tok, tools, cfg["memory"])
        ans = orch.run(q, max_steps=5)
        print(f"[Answer] {ans}")
