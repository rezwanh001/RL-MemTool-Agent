"""
Downloads HotpotQA and saves to arrow/parquet cached form.

Args:
    None
"""
from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("hotpot_qa", "distractor")
    ds.save_to_disk("data/hotpot_distractor")
    print("Saved to data/hotpot_distractor")
