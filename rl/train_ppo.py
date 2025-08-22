"""
PPO training loop using TRL with a text-only reward proxy.
We train the LM to produce better FINAL_ANSWER strings.
"""
import os
import json
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig
from rl.env import RLEnv
from agent.policy import LMPolicy


def load_records(path):
    recs = []
    with open(path, "r") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def main(cfg_path="configs/train.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Load tokenizer and base model
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cfg.get("use_qlora", False):
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype="float16")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            device_map="auto",
            quantization_config=bnb_cfg
        )
        base_model = prepare_model_for_kbit_training(base_model)
        peft_cfg = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        base_model = get_peft_model(base_model, peft_cfg)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            device_map="auto",
            torch_dtype="auto"
        )

    # Policy wrapper
    policy = LMPolicy(cfg["model_name"])
    policy.model = base_model
    policy.tok = tok

    # Load training data
    train_recs = load_records("data/processed/hotpot_train.json")

    # PPO config (removed model_name)
    ppo_cfg = PPOConfig(
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["train_batch_size"],
        mini_batch_size=cfg["mini_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        ppo_epochs=cfg["ppo_epochs"],
        target_kl=cfg["target_kl"],
        optimize_cuda_cache=True,
    )

    # PPO trainer
    trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy.model,
        tokenizer=tok
    )

    # Environment
    env = RLEnv(train_recs, policy, tok, cfg)

    # Training loop
    for step, ex in enumerate(train_recs):
        tr = env.rollout(ex)

        # PPO expects lists of tensors
        query = tok(ex["question"], return_tensors="pt").to(policy.model.device)
        resp = tok(tr.response, return_tensors="pt").to(policy.model.device)

        trainer.step([query["input_ids"][0]], [resp["input_ids"][0]], [tr.reward])

        if step % 50 == 0:
            os.makedirs(cfg["output_dir"], exist_ok=True)
            policy.model.save_pretrained(cfg["output_dir"])
            tok.save_pretrained(cfg["output_dir"])
            print(f"Step {step} reward {tr.reward:.3f}")

    # Save final model
    os.makedirs(cfg["output_dir"], exist_ok=True)
    policy.model.save_pretrained(cfg["output_dir"])
    tok.save_pretrained(cfg["output_dir"])
    print("✅ Training complete.")


if __name__ == "__main__":
    main()
