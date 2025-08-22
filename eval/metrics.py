"""
Computes EM, F1, Tool Use Rate, Avg Tools/Query, Steps, and simple latency proxy.
"""
from rl.rewards import exact_match, f1_score
import numpy as np

def aggregate(results):
    em = np.mean([exact_match(r["pred"], r["gold"]) for r in results])
    f1 = np.mean([f1_score(r["pred"], r["gold"]) for r in results])
    tool_rate = np.mean([1.0 if r["tool_calls"]>0 else 0.0 for r in results])
    avg_tools = np.mean([r["tool_calls"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    return {
        "EM": em, "F1": f1,
        "ToolUseRate": tool_rate,
        "AvgTools": avg_tools,
        "AvgSteps": avg_steps
    }
