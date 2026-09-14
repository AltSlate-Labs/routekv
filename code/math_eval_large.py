"""RouteKV validation — LARGE math eval with paired bootstrap CIs (GPU 1).
Shared prefix = fixed 4 GSM8K worked demos (truly shared across eval). Target = math specialist
answering the held-out question using a demo-prefix cache from elsewhere. 500 FRESH GSM8K test
(skip 80 dev). Per-example EM stored; paired 95% bootstrap CI of (condition - native) per direction.
"""
import torch, re, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; N_CAL,N_EVAL,DEV_SKIP,NDEMO,LAM=40,500,80,4,10.0; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/qa",adapter_name="qa"); model.load_adapter("adapters/math",adapter_name="math"); model.eval()
NL,NH,HD=base.config.num_hidden_layers,base.config.num_key_value_heads,base.config.head_dim
def rot(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def rope(nn):
    p=torch.arange(nn,device=dev).unsqueeze(0); hid=torch.zeros(1,nn,base.config.hidden_size,device=dev,dtype=torch.bfloat16)
    c,s=base.model.rotary_emb(hid,p); return c.unsqueeze(1).float(),s.unsqueeze(1).float()
def strip(k,c,s): return k*c-rot(k)*s
def app(k,c,s): return k*c+rot(k)*s
def kv(which,ids):
    if which is None:
        with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    else:
        model.set_adapter(which)
        with torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return (torch.stack([pkv.layers[l].keys[0] for l in range(NL)]).float(),torch.stack([pkv.layers[l].values[0] for l in range(NL)]).float())
def build_cache(K,V):
    c=DynamicCache()
    for l in range(NL): c.update(K[l].unsqueeze(0).to(torch.bfloat16),V[l].unsqueeze(0).to(torch.bfloat16),l)
    return c
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
# fixed shared demo prefix
demos="".join(f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n" for ex in [next(tr) for _ in range(NDEMO)])
demo_ids=tok(demos,return_tensors="pt").input_ids.cuda(); DP=demo_ids.shape[1]
cs,sn=rope(DP)

# fit base->math, qa->math ridge on demo-style prefixes (train)
S={(s,m):dict(Sxx=torch.zeros(NL,NH,HD,HD,device=dev),Sxy=torch.zeros(NL,NH,HD,HD,device=dev),sx=torch.zeros(NL,NH,HD,device=dev),sy=torch.zeros(NL,NH,HD,device=dev),n=0) for s in ["base","qa"] for m in "KV"}
for i in range(N_CAL):
    exs=[next(tr) for _ in range(NDEMO)]; pref="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in exs)
    ids=tok(pref,return_tensors="pt").input_ids[:,:DP].cuda(); P=ids.shape[1]; c2,s2=rope(P)
    tK,tV=kv("math",ids); tK=strip(tK,c2,s2)
    for sname in ["base","qa"]:
        sK,sV=kv(None if sname=="base" else "qa",ids); sK=strip(sK,c2,s2)
        for m,X,Y in [("K",sK,tK),("V",sV,tV)]:
            st=S[(sname,m)]; st["Sxx"]+=torch.einsum('lhsd,lhse->lhde',X,X); st["Sxy"]+=torch.einsum('lhsd,lhse->lhde',X,Y); st["sx"]+=X.sum(2); st["sy"]+=Y.sum(2); st["n"]+=P
W={};b={}
for (sname,m),st in S.items():
    n=st["n"]; Cxx=st["Sxx"]-torch.einsum('lhd,lhe->lhde',st["sx"],st["sx"])/n; Cxy=st["Sxy"]-torch.einsum('lhd,lhe->lhde',st["sx"],st["sy"])/n
    W[(sname,m)]=torch.linalg.solve(Cxx+LAM*torch.eye(HD,device=dev),Cxy); b[(sname,m)]=st["sy"]/n-torch.einsum('lhde,lhd->lhe',W[(sname,m)],st["sx"]/n)
def translate(sname,K,V,P):
    c2,s2=rope(P); Ksf=strip(K,c2,s2)
    Kh=torch.einsum('lhde,lhsd->lhse',W[(sname,"K")],Ksf)+b[(sname,"K")].unsqueeze(2); Vh=torch.einsum('lhde,lhsd->lhse',W[(sname,"V")],V)+b[(sname,"V")].unsqueeze(2)
    return app(Kh,c2,s2),Vh
print("fitted",flush=True)
def greedy(cache,primer,maxnew=200):
    model.set_adapter("math")
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(maxnew):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks)
conds=["native","base_direct","base_ridge","qa_direct","qa_ridge"]; per={c:[] for c in conds}
te=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))
for _ in range(DEV_SKIP): next(te)
# precompute source demo-prefix caches once (shared prefix is fixed)
srcK={}; srcV={}
for sname,which in [("base",None),("qa","qa")]:
    srcK[sname],srcV[sname]=kv(which,demo_ids)
mK,mV=kv("math",demo_ids)  # math's own demo cache (parity-ish native uses full)
for i in range(N_EVAL):
    ex=next(te); gold=lastnum(ex["answer"]); qids=tok(f"Question: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda()
    a={}
    model.set_adapter("math"); a["native"]=greedy(DynamicCache(),torch.cat([demo_ids,qids],1))
    a["base_direct"]=greedy(build_cache(srcK["base"],srcV["base"]),qids)
    tK,tV=translate("base",srcK["base"],srcV["base"],DP); a["base_ridge"]=greedy(build_cache(tK,tV),qids)
    a["qa_direct"]=greedy(build_cache(srcK["qa"],srcV["qa"]),qids)
    tK,tV=translate("qa",srcK["qa"],srcV["qa"],DP); a["qa_ridge"]=greedy(build_cache(tK,tV),qids)
    for c in conds: per[c].append(float(lastnum(a[c])==gold))
    if (i+1)%100==0: print(f"  {i+1}/{N_EVAL}",flush=True)
per={c:np.array(v) for c,v in per.items()}
def ci(d,B=5000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return np.percentile(m,2.5),np.percentile(m,97.5)
print(f"\n=== LARGE math eval (n={N_EVAL} fresh, math target, shared {NDEMO}-demo prefix) ===",flush=True)
nat=per["native"].mean(); print(f"native EM = {nat*100:.1f}%",flush=True)
for c in conds[1:]:
    d=per[c]-per["native"]; lo,hi=ci(d)
    print(f"{c:>11}: EM {per[c].mean()*100:.1f}%  ret {per[c].mean()/nat*100:.0f}%  | Δ {d.mean()*100:+.1f}pp 95%CI[{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
print("MATH_LARGE_DONE",flush=True)
