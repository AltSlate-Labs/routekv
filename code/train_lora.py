"""RouteKV step 2 — train a same-base LoRA specialist (QA or math) on Qwen3-1.7B.
Targets q/k/v/o so the adapter genuinely changes KV (required for a real handoff test).
Loss on completion tokens only. Usage: python train_lora.py --task qa|math --out DIR --steps N
"""
import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BACKBONE = "Qwen/Qwen3-1.7B"
MAXLEN = 1024
dev = "cuda"

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, choices=["qa", "math"])
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.bfloat16).cuda()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
model.train()

if a.task == "qa":
    ds = iter(load_dataset("hotpotqa/hotpot_qa", "distractor", split="train", streaming=True))
    def example():
        ex = next(ds)
        passages = "".join(" ".join(s) for s in ex["context"]["sentences"])
        passages = tok.decode(tok(passages, add_special_tokens=False).input_ids[:700])  # cap so answer survives
        return f"{passages}\n\nQuestion: {ex['question']}\nAnswer:", " " + ex["answer"]
else:
    ds = iter(load_dataset("openai/gsm8k", "main", split="train", streaming=True))
    def example():
        ex = next(ds)
        return f"Question: {ex['question']}\nAnswer:", " " + ex["answer"]

def batch(bs=1):
    seqs = []
    for _ in range(bs):
        p, c = example()
        pid = tok(p, add_special_tokens=False).input_ids
        cid = tok(c, add_special_tokens=False).input_ids + [tok.eos_token_id]
        pid = pid[:MAXLEN - len(cid)]                      # reserve room; completion always kept
        ids = pid + cid
        lab = [-100]*len(pid) + cid
        seqs.append((ids, lab))
    return seqs

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
import math
t0 = 0
for step in range(a.steps):
    opt.zero_grad()
    loss_acc = 0.0
    for _ in range(a.accum):
        ids, lab = batch(1)[0]
        x = torch.tensor([ids], device=dev); y = torch.tensor([lab], device=dev)
        out = model(input_ids=x, labels=y)
        (out.loss / a.accum).backward(); loss_acc += out.loss.item()/a.accum
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    if step % 25 == 0:
        print(f"[{a.task}] step {step}/{a.steps} loss {loss_acc:.4f}", flush=True)

model.save_pretrained(a.out)
print(f"SAVED {a.task} adapter -> {a.out}", flush=True)
print("LORA_DONE", flush=True)
