# agent/orchestration.py
"""
Simple step loop:
- Give LM system+memory+question
- If TOOL JSON returned: run tool, add EVIDENCE to memory
- Else: treat text as THINK notes; if FINAL_ANSWER found, stop
"""
import re, json
from .prompts import SYSTEM_PROMPT
from .memory_manager import MemoryManager

FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.+)", re.I)

class Orchestrator:
    def __init__(self, policy, tokenizer, tools: dict, mem_cfg: dict):
        self.policy = policy
        self.tools = tools
        self.memory = MemoryManager(
            budget_tokens=mem_cfg["budget_tokens"],
            tokenizer=tokenizer,
            priority_keys=mem_cfg.get("priority_keys", [])
        )

    def run(self, question: str, max_steps=5):
        for step in range(max_steps):
            reply = self.policy.step(SYSTEM_PROMPT, self.memory.snapshot(), question)
            tool_call = self.policy.try_parse_tool(reply)
            if tool_call:
                action = tool_call.get("action")
                args = tool_call.get("args", {})
                tool = self.tools.get(action)
                if tool:
                    res = tool.call(**args)
                    self.memory.add("evidence", res)
                else:
                    self.memory.add("note", f"Unknown tool: {action}")
            else:
                self.memory.add("note", reply)
                m = FINAL_RE.search(reply)
                if m:
                    return m.group(1).strip()
        return "I could not derive a confident answer within step limit."
