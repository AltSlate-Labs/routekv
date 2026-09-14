"""Generation-budget diagnostic: does the GSM8K reuse penalty shrink at a larger, FROZEN budget?
Native vs full-prefix reuse on the SAME 500 untouched examples (test[580:1080]), budget FROZEN at 320 tokens
(2x the 160 used in the main run). If the Δ shrinks toward the main run's −4.6, truncation contributed; if
it persists, truncation is not the explanation. Reports EM, cap-hit%, and paired Δ.
  python budget_diag.py 320
"""
import sys, re, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BUDGET=int(sys.argv[1]); N=500; NDEMO=4; dev="cuda"
BACKBONE="Qwen/Qwen3-1.7B"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/math",adapter_name="math"); model.eval()
NL=base.config.num_hidden_layers
INSTR="You are a careful math tutor. Read the problem, reason step by step, and end with the final numeric answer.\n\n"
tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
DEMOS="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
pre=tok(INSTR+DEMOS,add_special_tokens=False).input_ids

def base_kv(ids):
    with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return [(pkv.layers[l].keys,pkv.layers[l].values) for l in range(NL)]
def slice_cache(bkv,pb):
    c=DynamicCache()
    for l,(k,v) in enumerate(bkv): c.update(k[:,:,:pb,:].clone(),v[:,:,:pb,:].clone(),l)
    return c
def greedy(cache,primer):
    model.set_adapter("math")
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(BUDGET):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks), len(toks)>=BUDGET
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

it=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))
for _ in range(580): next(it)
nat=[]; reu=[]; capn=[]; capr=[]
for i in range(N):
    ex=next(it); gold=lastnum(ex["answer"])
    q=tok(f"Question: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
    full=torch.tensor([pre+q],device=dev); T=full.shape[1]
    a,cn=greedy(DynamicCache(),full); nat.append(float(lastnum(a)==gold)); capn.append(float(cn))
    bkv=base_kv(full); a,cr=greedy(slice_cache(bkv,T-1),full[:,T-1:]); reu.append(float(lastnum(a)==gold)); capr.append(float(cr))
    if (i+1)%100==0: print(f"  budget={BUDGET} {i+1}/{N}",flush=True)

nat=np.array(nat); reu=np.array(reu); d=reu-nat
idx=np.random.randint(0,N,(5000,N)); m=d[idx].mean(1); lo,hi=np.percentile(m,2.5),np.percentile(m,97.5)
print(f"\n=== generation-budget diagnostic (GSM8K test[580:1080], n={N}, budget={BUDGET} tok) ===",flush=True)
print(f"native EM {nat.mean()*100:.1f}  (cap-hit {np.mean(capn)*100:.1f}%)",flush=True)
print(f"reuse  EM {reu.mean()*100:.1f}  (cap-hit {np.mean(capr)*100:.1f}%)",flush=True)
print(f"reuse - native = {d.mean()*100:+.1f}pp  CI[{lo*100:+.1f},{hi*100:+.1f}]  (main run @160 was -4.6 [-8.8,-0.4], caps 15/30%)",flush=True)
np.savez(f"budget_{BUDGET}.npz", nat=nat, reu=reu, capn=np.array(capn), capr=np.array(capr))
print("BUDGET_DONE",flush=True)
