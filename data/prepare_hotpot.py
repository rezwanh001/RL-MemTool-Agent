from datasets import load_dataset
import os

# Load HotpotQA dataset
print("Loading HotpotQA dataset...")
ds = load_dataset("hotpot_qa", "fullwiki")

def _map(example):
    context_text = ""
    for pair in example["context"]:
        if len(pair) >= 2:
            title = pair[0]
            sents = pair[1]
            context_text += f"{title}: {' '.join(sents)} "
    return {
        "id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "context": context_text.strip()
    }


# Apply mapping
print("Processing train and validation splits...")
train = ds["train"].map(_map, remove_columns=ds["train"].column_names)
valid = ds["validation"].map(_map, remove_columns=ds["validation"].column_names)

# Save processed dataset
# os.makedirs("data/processed", exist_ok=True)
# train.to_json("data/processed/hotpot_train.json")
# valid.to_json("data/processed/hotpot_valid.json")

os.makedirs("data/processed", exist_ok=True)
train.to_json("data/processed/hotpot_train.json")
valid.to_json("data/processed/hotpot_valid.json")

print("✅ Saved processed HotpotQA data to data/processed/")
