"""
No tools, no memory manager. Direct answer generation baseline.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

def answer(model_name, question):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    prompt = f"Answer concisely:\n{question}\n"
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=128)
    return tok.decode(out[0], skip_special_tokens=True)

# Use eval/evaluate.py pattern to score on validation
