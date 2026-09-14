"""RouteKV diagnostic 1 follow-ups (refinements 1 & 2 requested):
 (1) DIRECT difference test: bootstrap [(math base-reuse penalty) - (qa base-reuse penalty)] on the SAME
     math examples, plus each adapter's own penalty CI and the reuse-advantage CI. Answers whether the two
     adapters' reuse penalties actually differ (vs. the interaction being asserted from 1 sig + 3 nonsig cells).
 (2) PARITY inspection: own-KV parity sat +0.9pp above native for both adapters (~1 answer/120). Dump the
     examples where parity disagrees with native, to check for a harness contribution before ruling it out.
Reproduces diag_matrix.py's math setup EXACTLY: first NDEMO=4 gsm8k train as shared demos, first N=120 test.
"""
import torch, re, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; N=120; NDEMO=4; MX=160; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/qa",adapter_name="qa"); model.load_adapter("adapters/math",adapter_name="math"); model.eval()
NL=base.config.num_hidden_layers
def kv(which,ids):
    if which is None:
        with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    else:
        model.set_adapter(which)
        with torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return [(pkv.layers[l].keys,pkv.layers[l].values) for l in range(NL)]
def build_cache(kvp):
    c=DynamicCache()
    for l,(k,v) in enumerate(kvp): c.update(k.clone(),v.clone(),l)
    return c
def greedy(target,cache,primer):
    model.set_adapter(target)
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(MX):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks)
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
demos="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
demo_ids=tok(demos,return_tensors="pt").input_ids.cuda()
it=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))

S={f"{t}_{c}":[] for t in ["math","qa"] for c in ["native","parity","base"]}
disc=[]  # parity!=native cases (math adapter)
bK_math_pref=kv(None,demo_ids)  # base demo-KV reused across all examples' question suffixes
for i in range(N):
    ex=next(it); gold=lastnum(ex["answer"]); q=ex["question"]
    qids=tok(f"Question: {q}\nAnswer:",return_tensors="pt").input_ids.cuda()
    for t in ["math","qa"]:
        a_nat=greedy(t,DynamicCache(),torch.cat([demo_ids,qids],1)); S[f"{t}_native"].append(float(lastnum(a_nat)==gold))
        tK=kv(t,demo_ids); a_par=greedy(t,build_cache(tK),qids); S[f"{t}_parity"].append(float(lastnum(a_par)==gold))
        a_bas=greedy(t,build_cache(bK_math_pref),qids); S[f"{t}_base"].append(float(lastnum(a_bas)==gold))
        if t=="math" and (lastnum(a_nat)==gold)!=(lastnum(a_par)==gold):
            disc.append((i,q[:90],gold,lastnum(a_nat),lastnum(a_par)))
    if (i+1)%40==0: print(f"  {i+1}/{N}",flush=True)

S={k:np.array(v) for k,v in S.items()}
def bci(d,B=5000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
pen_m=S["math_base"]-S["math_native"]; pen_q=S["qa_base"]-S["qa_native"]
diff=pen_m-pen_q
adv_nat=S["math_native"]-S["qa_native"]; adv_base=S["math_base"]-S["qa_base"]
print(f"\n=== Refinement 1: direct difference test (math task, n={N}, paired) ===",flush=True)
for nm,d in [("math-adapter reuse penalty (base-native)",pen_m),("qa-adapter reuse penalty (base-native)",pen_q),
             ("DIFFERENCE of penalties (math - qa)",diff),("math advantage NATIVE (math-qa)",adv_nat),
             ("math advantage under REUSE (math-qa)",adv_base)]:
    m,lo,hi=bci(d); print(f"  {nm:>42}: {m*100:+6.1f}pp  CI[{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
print(f"\n=== Refinement 2: parity vs native discrepancies (math adapter) ===",flush=True)
print(f"  math parity={S['math_parity'].mean()*100:.1f} native={S['math_native'].mean()*100:.1f} "
      f"n_disagree={len(disc)} (net {(S['math_parity'].mean()-S['math_native'].mean())*100:+.1f}pp)",flush=True)
for i,q,g,nn,pp in disc: print(f"  ex{i}: gold={g} native->{nn} parity->{pp} | {q}",flush=True)
print("DIFF_TEST_DONE",flush=True)
