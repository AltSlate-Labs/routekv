"""RouteKV residency test: two simultaneously-retained branches (QA + math) over ONE shared context.
Compares peak GPU memory for NATIVE (each branch prefills the context itself) vs REUSE (one base prefill,
both branches start from it), and checks whether prefix sharing PERSISTS as both branches generate.
Key question (reviewer): with transformers DynamicCache, does the shared prefix stay shared during
generation, or does concatenation copy it per branch (breaking the saving)?
  python residency.py 8192 32 20
"""
import sys, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

L=int(sys.argv[1]); G=int(sys.argv[2]); N=int(sys.argv[3]); dev="cuda"
BACKBONE="Qwen/Qwen3-1.7B"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/qa",adapter_name="qa"); model.load_adapter("adapters/math",adapter_name="math"); model.eval()
NL=base.config.num_hidden_layers; WEIGHTS=torch.cuda.memory_allocated()

def prefill(adapter, ids):
    if adapter is None:
        with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    else:
        model.set_adapter(adapter)
        with torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return pkv
def cache_from(pkv, share):
    c=DynamicCache()
    for l in range(NL):
        k,v=pkv.layers[l].keys,pkv.layers[l].values
        c.update(k if share else k.clone(), v if share else v.clone(), l)
    return c
def gen_step(adapter, cache, last):
    model.set_adapter(adapter)
    with torch.inference_mode(): out=model(input_ids=last,past_key_values=cache,use_cache=True)
    return int(out.logits[0,-1].argmax()), out.past_key_values

# build one L-token QA context (gold-preserving construction, same as matched.py)
it=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
pool_it=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
POOL=[]
for _ in range(60):
    ex=next(pool_it); POOL+= [" ".join(s) for s in ex["context"]["sentences"]]
for _ in range(600): next(it)
def build_ctx():
    ex=next(it); gt=set(ex["supporting_facts"]["title"]); ps=list(zip(ex["context"]["title"],ex["context"]["sentences"]))
    chosen=[" ".join(s) for t,s in ps if t in gt]+[" ".join(s) for t,s in ps if t not in gt]
    p=0
    while len(tok(" ".join(chosen),add_special_tokens=False).input_ids)<L: chosen.append(POOL[p%len(POOL)]); p+=1
    cids=tok(" ".join(chosen),add_special_tokens=False).input_ids[:L]
    q=tok(f"\n\nQuestion: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
    return torch.tensor([cids+q],device=dev)

pk_nat=[]; pk_reuse=[]; share_before=[]; share_after=[]
for i in range(N):
    full=build_ctx(); T=full.shape[1]
    # NATIVE: each branch prefills the full context with its own adapter; both retained; interleave-generate
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cA=cache_from(prefill("qa",full),share=False); cB=cache_from(prefill("math",full),share=False)
    lastA=full[:,-1:].clone(); lastB=full[:,-1:].clone()
    for _ in range(G):
        nA,cA=gen_step("qa",cA,lastA); lastA=torch.tensor([[nA]],device=dev)
        nB,cB=gen_step("math",cB,lastB); lastB=torch.tensor([[nB]],device=dev)
    pk_nat.append(torch.cuda.max_memory_allocated()/1e9)
    del cA,cB

    # REUSE: one base prefill; both branches START from it sharing tensors (no clone); interleave-generate
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    bkv=prefill(None,full)
    cA=cache_from(bkv,share=True); cB=cache_from(bkv,share=True)
    # sharing BEFORE generation: do the two branch caches point at the same prefix tensor?
    sb=(cA.layers[0].keys.data_ptr()==cB.layers[0].keys.data_ptr()==bkv.layers[0].keys.data_ptr())
    # process each branch's (empty here) suffix by generating; prefix gets cat'd on first step
    lastA=full[:,-1:].clone(); lastB=full[:,-1:].clone()
    nA,cA=gen_step("qa",cA,lastA); lastA=torch.tensor([[nA]],device=dev)
    nB,cB=gen_step("math",cB,lastB); lastB=torch.tensor([[nB]],device=dev)
    # sharing AFTER first generation step: still the same prefix memory?
    sa=(cA.layers[0].keys.data_ptr()==cB.layers[0].keys.data_ptr())
    for _ in range(G-1):
        nA,cA=gen_step("qa",cA,lastA); lastA=torch.tensor([[nA]],device=dev)
        nB,cB=gen_step("math",cB,lastB); lastB=torch.tensor([[nB]],device=dev)
    pk_reuse.append(torch.cuda.max_memory_allocated()/1e9)
    share_before.append(float(sb)); share_after.append(float(sa))
    del cA,cB,bkv
    if (i+1)%5==0: print(f"  L={L} {i+1}/{N}",flush=True)

pk_nat=np.array(pk_nat); pk_reuse=np.array(pk_reuse)
print(f"\n=== residency: two branches (QA+math) over one shared context, L={L}, G={G}, n={N} ===",flush=True)
print(f"context tokens ≈ {T}; weights baseline {WEIGHTS/1e9:.2f} GB",flush=True)
print(f"peak GPU memory  NATIVE (2 independent prefills): {pk_nat.mean():.2f} GB",flush=True)
print(f"peak GPU memory  REUSE  (1 base prefill, shared): {pk_reuse.mean():.2f} GB",flush=True)
print(f"reuse / native peak = {pk_reuse.mean()/pk_nat.mean():.2f}x   (saving would be <1.0)",flush=True)
print(f"prefix sharing BEFORE generation: {np.mean(share_before)*100:.0f}% of trials shared",flush=True)
print(f"prefix sharing AFTER first gen step: {np.mean(share_after)*100:.0f}% of trials shared",flush=True)
print("RESIDENCY_DONE",flush=True)
