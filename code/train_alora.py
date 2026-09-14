"""RouteKV validation — train a matched aLoRA QA specialist (GPU 2).
Activated LoRA: adapter applies only AFTER the invocation tokens ("Answer:"), so the
passages+question prefix uses base weights -> base-prefix KV reuse is structurally exact.
Same training data/budget as the standard QA LoRA, for a like-for-like comparison.
"""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; MAXLEN=1024; STEPS=800; ACCUM=8; LR=2e-4; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token=tok.eos_token
inv=tok("Answer:",add_special_tokens=False).input_ids   # activation boundary
print("invocation tokens:",inv,flush=True)
model=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda()
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM",
    alora_invocation_tokens=inv))
model.print_trainable_parameters(); model.train()
ds=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="train",streaming=True))
def ex():
    e=next(ds); p="".join(" ".join(s) for s in e["context"]["sentences"])
    p=tok.decode(tok(p,add_special_tokens=False).input_ids[:700])
    return f"{p}\n\nQuestion: {e['question']}\nAnswer:"," "+e["answer"]
def batch():
    p,c=ex(); pid=tok(p,add_special_tokens=False).input_ids; cid=tok(c,add_special_tokens=False).input_ids+[tok.eos_token_id]
    pid=pid[:MAXLEN-len(cid)]; return pid+cid,[-100]*len(pid)+cid
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=LR)
for step in range(STEPS):
    opt.zero_grad(); la=0.0
    for _ in range(ACCUM):
        ids,lab=batch(); x=torch.tensor([ids],device=dev); y=torch.tensor([lab],device=dev)
        out=model(input_ids=x,labels=y); (out.loss/ACCUM).backward(); la+=out.loss.item()/ACCUM
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0); opt.step()
    if step%25==0: print(f"[alora-qa] step {step}/{STEPS} loss {la:.4f}",flush=True)
model.save_pretrained("adapters/qa_alora"); print("SAVED alora-qa -> adapters/qa_alora",flush=True); print("ALORA_DONE",flush=True)
