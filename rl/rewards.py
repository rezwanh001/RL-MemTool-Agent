"""
Reward functions: EM, F1, tool penalties, long-think penalty.
"""
import re

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0

def f1_score(pred: str, gold: str) -> float:
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p or not g: return 0.0
    commons = {}
    for w in p:
        commons[w] = min(p.count(w), g.count(w))
    overlap = sum(commons.values())
    if overlap == 0: return 0.0
    prec = overlap / len(p)
    rec = overlap / len(g)
    return 2 * prec * rec / (prec + rec)

def reward(pred: str, gold: str, steps_used: int, tool_calls: int, cfg):
    em = exact_match(pred, gold)
    f1 = f1_score(pred, gold)
    r = cfg["em_weight"] * em + cfg["f1_weight"] * f1
    r -= cfg["tool_use_penalty"] * tool_calls
    r -= cfg["long_think_penalty"] * max(0, steps_used - 2)
    return r
