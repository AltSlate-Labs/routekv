"""RouteKV validation — LARGE QA eval with paired bootstrap CIs (GPU 0).
500 FRESH HotpotQA val examples (skip first 60 = dev). QA target answering with a shared-prefix
(passages) cache. Per-example F1 stored; paired 95% bootstrap CI of (condition - native) reported
separately per condition/direction. Narrow claim: does prefix reuse ~preserve task quality here?
"""
import torch, re, string, collections, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; N_CAL,N_EVAL,DEV_SKIP,MAXP,LAM=40,500,60,700,10.0; dev="cuda"
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
# fit base->QA, math->QA ridge
S={(s,m):dict(Sxx=torch.zeros(NL,NH,HD,HD,device=dev),Sxy=torch.zeros(NL,NH,HD,HD,device=dev),sx=torch.zeros(NL,NH,HD,device=dev),sy=torch.zeros(NL,NH,HD,device=dev),n=0) for s in ["base","math"] for m in "KV"}
tr=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="train",streaming=True))
for i in range(N_CAL):
    ex=next(tr); pas="".join(" ".join(x) for x in ex["context"]["sentences"]); ids=tok(pas,return_tensors="pt").input_ids[:,:MAXP].cuda(); P=ids.shape[1]; cs,sn=rope(P)
    qK,qV=kv("qa",ids); qK=strip(qK,cs,sn)
    for sname in ["base","math"]:
        sK,sV=kv(None if sname=="base" else "math",ids); sK=strip(sK,cs,sn)
        for m,X,Y in [("K",sK,qK),("V",sV,qV)]:
            st=S[(sname,m)]; st["Sxx"]+=torch.einsum('lhsd,lhse->lhde',X,X); st["Sxy"]+=torch.einsum('lhsd,lhse->lhde',X,Y); st["sx"]+=X.sum(2); st["sy"]+=Y.sum(2); st["n"]+=P
W={};b={}
for (sname,m),st in S.items():
    n=st["n"]; Cxx=st["Sxx"]-torch.einsum('lhd,lhe->lhde',st["sx"],st["sx"])/n; Cxy=st["Sxy"]-torch.einsum('lhd,lhe->lhde',st["sx"],st["sy"])/n
    W[(sname,m)]=torch.linalg.solve(Cxx+LAM*torch.eye(HD,device=dev),Cxy); b[(sname,m)]=st["sy"]/n-torch.einsum('lhde,lhd->lhe',W[(sname,m)],st["sx"]/n)
def translate(sname,K,V,P):
    cs,sn=rope(P); Ksf=strip(K,cs,sn)
    Kh=torch.einsum('lhde,lhsd->lhse',W[(sname,"K")],Ksf)+b[(sname,"K")].unsqueeze(2); Vh=torch.einsum('lhde,lhsd->lhse',W[(sname,"V")],V)+b[(sname,"V")].unsqueeze(2)
    return app(Kh,cs,sn),Vh
print("fitted",flush=True)
def greedy(cache,primer):
    model.set_adapter("qa")
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
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
conds=["native","base_direct","base_ridge","math_direct","math_ridge"]
per={c:[] for c in conds}
va=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
for _ in range(DEV_SKIP): next(va)
for i in range(N_EVAL):
    ex=next(va); gold=ex["answer"]; pas="".join(" ".join(x) for x in ex["context"]["sentences"])
    pids=tok(pas,return_tensors="pt").input_ids[:,:MAXP].cuda(); qids=tok(f"\n\nQuestion: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda(); P=pids.shape[1]
    a={}; model.set_adapter("qa"); a["native"]=greedy(DynamicCache(),torch.cat([pids,qids],1))
    bK,bV=kv(None,pids); a["base_direct"]=greedy(build_cache(bK,bV),qids)
    tK,tV=translate("base",bK,bV,P); a["base_ridge"]=greedy(build_cache(tK,tV),qids)
    mK,mV=kv("math",pids); a["math_direct"]=greedy(build_cache(mK,mV),qids)
    tK,tV=translate("math",mK,mV,P); a["math_ridge"]=greedy(build_cache(tK,tV),qids)
    for c in conds: per[c].append(f1(a[c],gold))
    if (i+1)%100==0: print(f"  {i+1}/{N_EVAL}",flush=True)
per={c:np.array(v) for c,v in per.items()}
def bootstrap_ci(diff,B=5000):
    idx=np.random.randint(0,len(diff),(B,len(diff))); means=diff[idx].mean(1); return np.percentile(means,2.5),np.percentile(means,97.5)
print(f"\n=== LARGE QA eval (n={N_EVAL} fresh, QA target, shared prefix) ===",flush=True)
nat=per["native"].mean()
print(f"native F1 = {nat*100:.1f}%",flush=True)
for c in conds[1:]:
    d=per[c]-per["native"]; lo,hi=bootstrap_ci(d)
    print(f"{c:>12}: F1 {per[c].mean()*100:.1f}%  ret {per[c].mean()/nat*100:.0f}%  | Δ(vs native) {d.mean()*100:+.1f}pp  95%CI[{lo*100:+.1f},{hi*100:+.1f}]pp",flush=True)
print("QA_LARGE_DONE",flush=True)
