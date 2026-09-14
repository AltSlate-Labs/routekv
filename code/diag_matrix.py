"""RouteKV diagnostic 1 — task x adapter matrix. Does reuse-degradation follow the TASK or the
ADAPTER? For each task (QA, math) and each target adapter (qa, math): native, parity (target's OWN
prefix-KV reused -> harness sanity), and base-reuse (base's prefix-KV reused). Gap = base-reuse - native.
"""
import torch, re, string, collections, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE="Qwen/Qwen3-1.7B"; N=120; MAXP=700; NDEMO=4; dev="cuda"
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
    return [ (pkv.layers[l].keys, pkv.layers[l].values) for l in range(NL) ]
def build_cache(kvpairs):
    c=DynamicCache()
    for l,(k,v) in enumerate(kvpairs): c.update(k.clone(),v.clone(),l)
    return c
def greedy(target,cache,primer,maxnew):
    model.set_adapter(target)
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(maxnew):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        if maxnew<=32 and "\n" in tok.decode(toks): break
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks)
def norm(s): s=s.lower(); s=re.sub(r'\b(a|an|the)\b',' ',s); s="".join(c for c in s if c not in string.punctuation); return " ".join(s.split())
def f1(p,g):
    p=p.strip().split("\n")[0]; pt,gt=norm(p).split(),norm(g).split()
    if not pt or not gt: return float(pt==gt)
    cm=collections.Counter(pt)&collections.Counter(gt); ns=sum(cm.values())
    if ns==0: return 0.0
    pr,rc=ns/len(pt),ns/len(gt); return 2*pr*rc/(pr+rc)
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

def run_task(task, targets):
    res={(t,c):[] for t in targets for c in ["native","parity","base"]}
    if task=="qa":
        it=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True)); mx=24
    else:
        tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
        demos="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
        demo_ids=tok(demos,return_tensors="pt").input_ids.cuda()
        it=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True)); mx=160
        base_pref=None
    for i in range(N):
        ex=next(it)
        if task=="qa":
            gold=ex["answer"]; pas="".join(" ".join(s) for s in ex["context"]["sentences"]); pas=tok.decode(tok(pas,add_special_tokens=False).input_ids[:MAXP])
            pids=tok(pas,return_tensors="pt").input_ids.cuda(); qids=tok(f"\n\nQuestion: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda()
            score=lambda a: f1(a,gold)
        else:
            gold=lastnum(ex["answer"]); pids=demo_ids; qids=tok(f"Question: {ex['question']}\nAnswer:",return_tensors="pt").input_ids.cuda()
            score=lambda a: float(lastnum(a)==gold)
        bK=kv(None,pids)
        for t in targets:
            # native: target prefills prefix+question
            a=greedy(t,DynamicCache(),torch.cat([pids,qids],1),mx); res[(t,"native")].append(score(a))
            # parity: target's own prefix-KV reused
            tK=kv(t,pids); a=greedy(t,build_cache(tK),qids,mx); res[(t,"parity")].append(score(a))
            # base-reuse: base's prefix-KV reused
            a=greedy(t,build_cache(bK),qids,mx); res[(t,"base")].append(score(a))
        if (i+1)%40==0: print(f"  {task} {i+1}/{N}",flush=True)
    return res

def ci(d,B=3000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return np.percentile(m,2.5),np.percentile(m,97.5)
print("=== Diagnostic 1: task x adapter matrix (metric = QA-F1 / math-EM) ===",flush=True)
for task,targets in [("qa",["qa","math"]),("math",["math","qa"])]:
    res=run_task(task,targets)
    print(f"\n[{task} task]",flush=True)
    for t in targets:
        nat=np.array(res[(t,"native")]); par=np.array(res[(t,"parity")]); bas=np.array(res[(t,"base")])
        d=bas-nat; lo,hi=ci(d)
        print(f"  target={t:>4}: native {nat.mean()*100:5.1f} | parity {par.mean()*100:5.1f} (own-KV, ~native) | base-reuse {bas.mean()*100:5.1f}  Δ {d.mean()*100:+.1f}pp CI[{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
print("DIAG_MATRIX_DONE",flush=True)
