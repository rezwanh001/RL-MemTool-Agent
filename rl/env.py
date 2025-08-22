"""
Gym-like environment wrapper for TRL PPO.
Generates prompts, runs a short orchestration rollout, returns text for PPO.
"""
import json, random
from dataclasses import dataclass
from agent.orchestration import Orchestrator
from agent.tools import RetrievalTool, WebSearchTool
from agent.policy import LMPolicy
from agent.prompts import SYSTEM_PROMPT
from .rewards import reward

@dataclass
class Transition:
    prompt: str
    response: str
    reward: float

class RLEnv:
    """
    Initializes the RL environment.

    Args:
        records (List[dict]): dataset records with 'question', 'gold_answer', 'candidate_contexts'
        policy (LMPolicy): language model wrapper
        cfg (dict): training config including tools and memory
    """
    def __init__(self, records, policy: LMPolicy, tok, cfg):
        self.records = records
        self.policy = policy
        self.tok = tok
        self.cfg = cfg

    def build_tools(self, candidate_contexts):
        tools = {}
        if self.cfg["tools"]["enable_retrieval"]:
            tools["retrieve"] = RetrievalTool(candidate_contexts)
        if self.cfg["tools"]["enable_web_search"]:
            tools["web_search"] = WebSearchTool()
        return tools

    def rollout(self, example):
        tools = self.build_tools(example["candidate_contexts"])
        orch = Orchestrator(self.policy, self.tok, tools, self.cfg["memory"])
        answer = orch.run(example["question"], max_steps=5)
        # For logging, we can approximate steps/tool_calls from memory contents
        steps_used = len(orch.memory.notes)
        tool_calls = sum(1 for n in orch.memory.notes if n.kind == "evidence")
        r = reward(answer, example["gold_answer"], steps_used, tool_calls, self.cfg["rewards"])
        return Transition(prompt=example["question"], response=answer, reward=r)
