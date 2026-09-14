"""RouteKV diagnostic 2 - reuse-boundary (takeover) sweep on the math task.
3-segment prompt: [instructions | demonstrations | question] -> answer. Full prompt + decoding held
identical; only the boundary between reused base-KV and math-adapter computation moves:
  native      : reuse none          | math computes entire prompt
  early       : reuse instructions  | math computes demos + question
  question    : reuse instr + demos | math computes question
  full_prefix : reuse whole prompt  | math computes answer only
Answers: does making the math specialist encode the question itself restore quality? -> memory rule.
Reports paired EM Δ vs native, reused-cache bytes, and adapter-computed prompt tokens (compute proxy).
"""
import torch, re, time, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; N=120; NDEMO=4; MAXNEW=160; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/math",adapter_name="math"); model.eval()
NL,NH,HD=base.config.num_hidden_layers,base.config.num_key_value_heads,base.config.head_dim

INSTR="You are a careful math tutor. Read the problem, reason step by step, and end with the final numeric answer.\n\n"
tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
DEMOS="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
instr_ids=tok(INSTR,add_special_tokens=False).input_ids
demo_ids=tok(DEMOS,add_special_tokens=False).input_ids
L_INSTR=len(instr_ids); L_DEMO=len(demo_ids)

def base_kv(ids):
    with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return [(pkv.layers[l].keys, pkv.layers[l].values) for l in range(NL)]
def cache_slice(bkv,pb):
    c=DynamicCache()
    for l,(k,v) in enumerate(bkv): c.update(k[:,:,:pb,:].clone(),v[:,:,:pb,:].clone(),l)
    return c
def greedy(cache,primer):
    model.set_adapter("math")
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(MAXNEW):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks)
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""
def bytes_of(pb): return pb*NL*NH*HD*2*2  # K+V, bf16

conds=["native","early","question","full_prefix"]
per={c:[] for c in conds}; reused_tok={c:0 for c in conds}; comp_tok={c:0 for c in conds}
te=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))
for _ in range(80): next(te)  # skip dev
for i in range(N):
    ex=next(te); gold=lastnum(ex["answer"])
    q_ids=tok(f"Question: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
    full=torch.tensor([instr_ids+demo_ids+q_ids],device=dev); T=full.shape[1]
    bkv=base_kv(full)
    bounds={"native":0,"early":L_INSTR,"question":L_INSTR+L_DEMO,"full_prefix":T-1}
    for c in conds:
        pb=bounds[c]
        cache=DynamicCache() if pb==0 else cache_slice(bkv,pb)
        primer=full if pb==0 else full[:,pb:]
        a=greedy(cache,primer)
        per[c].append(float(lastnum(a)==gold)); reused_tok[c]=pb; comp_tok[c]=T-pb
    if (i+1)%40==0: print(f"  {i+1}/{N}",flush=True)

def ci(d,B=3000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return np.percentile(m,2.5),np.percentile(m,97.5)
per={c:np.array(v) for c,v in per.items()}; nat=per["native"].mean()
print(f"\n=== Diagnostic 2: takeover-boundary sweep (math task, n={N}) ===",flush=True)
print(f"segment tokens: instructions={L_INSTR}, demos={L_DEMO}, +question(var). native EM={nat*100:.1f}%",flush=True)
print(f"{'condition':>12} {'EM':>6} {'Δ vs native':>22} {'reused_tok':>11} {'reused_KV_MB':>12} {'adapter_computes_tok':>20}",flush=True)
for c in conds:
    d=per[c]-per["native"]; lo,hi=ci(d)
    dd="" if c=="native" else f"{d.mean()*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]"
    print(f"{c:>12} {per[c].mean()*100:5.1f} {dd:>22} {reused_tok[c]:>11} {bytes_of(reused_tok[c])/1e6:>11.1f} {comp_tok[c]:>20}",flush=True)
print("BOUNDARY_DONE",flush=True)
