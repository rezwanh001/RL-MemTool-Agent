"""
Wraps the HF LM to produce next-step text given system+memory+user.
Also provides a simple action decoder that detects TOOL JSON.
"""
import json, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TOOL_RE = re.compile(r"\{[\s\S]*\"action\"\s*:\s*\"([^\"]+)\"[\s\S]*\}")

class LMPolicy:
    """
    Initializes the policy model and tokenizer.
    """
    def __init__(self, model_name: str, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.device = device

    @torch.no_grad()
    def step(self, system: str, memory: str, question: str, max_new_tokens: int = 256):
        prompt = f"<|system|>\n{system}\n<|memory|>\n{memory}\n<|user|>\n{question}\n<|assistant|>\n"
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        reply = text[len(self.tok.decode(inputs['input_ids'][0], skip_special_tokens=True)):]
        return reply.strip()

    @staticmethod
    def try_parse_tool(reply: str):
        m = TOOL_RE.search(reply)
        if not m:
            return None
        try:
            blob = json.loads(m.group(0))
            return blob
        except Exception:
            return None
