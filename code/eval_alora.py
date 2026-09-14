"""RouteKV validation — aLoRA eval (GPU 2). aLoRA applies only AFTER the invocation ("Answer:"),
so the passages+question PREFIX KV must equal the BASE's -> base-prefix reuse is STRUCTURALLY EXACT
(not approximate). (1) verify prefix KV(base) == KV(aLoRA) numerically; (2) aLoRA native F1 vs the
standard-LoRA approach (native 53.0 / direct-reuse 53.3 from qa_eval_large).
"""
import torch, re, string, collections, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; NCHK,NEVAL,MAXP=20,200,700; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/qa_alora",adapter_name="alora"); model.eval()
NL=base.config.num_hidden_layers
def kv(which,ids):
    if which is None:
        with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    else:
        model.set_adapter(which)
        with torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return torch.stack([pkv.layers[l].keys[0] for l in range(NL)]).float(), torch.stack([pkv.layers[l].values[0] for l in range(NL)]).float()

# (1) structural check: prefix KV base vs aLoRA (prefix ends at the invocation "Answer:")
va=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
maxdiff=0.0
for _ in range(NCHK):
    ex=next(va); pas="".join(" ".join(s) for s in ex["context"]["sentences"]); pas=tok.decode(tok(pas,add_special_tokens=False).input_ids[:MAXP])
    ids=tok(f"{pas}\n\nQuestion: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda()
    bK,bV=kv(None,ids); aK,aV=kv("alora",ids)
    maxdiff=max(maxdiff, (bK-aK).abs().max().item(), (bV-aV).abs().max().item())
print(f"[structural] max |KV(base) - KV(aLoRA)| over prefix (pre-invocation) = {maxdiff:.2e}  (≈0 ⇒ base-prefix reuse is EXACT)",flush=True)

# (2) aLoRA native F1
def gen(ids):
    model.set_adapter("alora")
    with torch.inference_mode(): out=model(input_ids=ids,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(24):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        if "\n" in tok.decode(toks): break
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks).strip().split("\n")[0].strip()
def norm(s): s=s.lower(); s=re.sub(r'\b(a|an|the)\b',' ',s); s="".join(c for c in s if c not in string.punctuation); return " ".join(s.split())
def f1(p,g):
    pt,gt=norm(p).split(),norm(g).split()
    if not pt or not gt: return float(pt==gt)
    cm=collections.Counter(pt)&collections.Counter(gt); ns=sum(cm.values())
    if ns==0: return 0.0
    pr,rc=ns/len(pt),ns/len(gt); return 2*pr*rc/(pr+rc)
scores=[]
for _ in range(NEVAL):
    ex=next(va); gold=ex["answer"]; pas="".join(" ".join(s) for s in ex["context"]["sentences"]); pas=tok.decode(tok(pas,add_special_tokens=False).input_ids[:MAXP])
    ids=tok(f"{pas}\n\nQuestion: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda()
    scores.append(f1(gen(ids),gold))
scores=np.array(scores)
print(f"\n=== aLoRA eval (n={NEVAL}) ===",flush=True)
print(f"aLoRA native F1 = {scores.mean()*100:.1f}%  (standard-LoRA ref: native 53.0 / direct-reuse 53.3)",flush=True)
print(f"structural exact base-prefix reuse ⇒ cached == uncached by construction (max diff {maxdiff:.1e})",flush=True)
print("ALORA_EVAL_DONE",flush=True)
