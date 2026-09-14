"""RouteKV matched experiment (frozen protocol). One task+bucket per process:
  python matched.py gsm8k 0 500        # GSM8K regression, natural ~655 tok, test[580:1080]
  python matched.py qa 2000 300        # supplied-context QA, ~2K constructed context
  python matched.py qa 8000 200        # supplied-context QA, ~8K constructed context
Three conditions on IDENTICAL examples/prompts: base-only, native adapter, full-prefix base-KV reuse.
Records per example: quality (EM/F1), TTFT, completion latency, peak GPU mem (abs + over-weights), cap-hit.
Paired bootstrap CIs of (condition - native). Saves per-example arrays to matched_{task}_{bucket}.npz.
QA long contexts: gold (supporting) paragraphs ALWAYS kept; distractors appended (from the example, then a
fixed disjoint pool) to reach the target length; final distractor truncated to hit length, gold never cut.
Untouched examples: GSM8K test[580:1080]; QA validation index >=600; QA distractor pool from validation[0:60].
"""
import sys, time, re, string, collections, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

TASK=sys.argv[1]; TGT=int(sys.argv[2]); N=int(sys.argv[3]); dev="cuda"
BACKBONE="Qwen/Qwen3-1.7B"; NDEMO=4
MAXNEW=160 if TASK=="gsm8k" else 48
ADAP="math" if TASK=="gsm8k" else "qa"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,f"adapters/{ADAP}",adapter_name=ADAP); model.eval()
NL=base.config.num_hidden_layers
WEIGHTS=torch.cuda.memory_allocated()  # resident weights baseline

def base_kv(ids):
    with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return [(pkv.layers[l].keys,pkv.layers[l].values) for l in range(NL)]
def slice_cache(bkv,pb):
    c=DynamicCache()
    for l,(k,v) in enumerate(bkv): c.update(k[:,:,:pb,:].clone(),v[:,:,:pb,:].clone(),l)
    return c
def sync(): torch.cuda.synchronize()

def run(cond, full, bkv=None):
    """cond in {base,native,reuse}. Returns (text, ttft_ms, total_ms, peak_abs_gb, peak_over_gb, capped)."""
    T=full.shape[1]
    torch.cuda.reset_peak_memory_stats()
    if cond=="reuse":
        pb=T-1; cache=slice_cache(bkv,pb); primer=full[:,pb:]
    else:
        cache=DynamicCache(); primer=full
    sync(); t0=time.perf_counter()
    with torch.inference_mode():
        if cond=="base":
            with model.disable_adapter(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
        else:
            model.set_adapter(ADAP); out=model(input_ids=primer,past_key_values=cache,use_cache=True)
        lg=out.logits[0,-1]; cache=out.past_key_values
        nt=int(lg.argmax()); sync(); ttft=(time.perf_counter()-t0)*1000
        toks=[]
        for _ in range(MAXNEW):
            if nt==tok.eos_token_id: break
            toks.append(nt)
            with (model.disable_adapter() if cond=="base" else torch.inference_mode()):
                out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True)
            lg=out.logits[0,-1]; cache=out.past_key_values; nt=int(lg.argmax())
    sync(); total=(time.perf_counter()-t0)*1000
    peak=torch.cuda.max_memory_allocated()
    capped=len(toks)>=MAXNEW
    return tok.decode(toks), ttft, total, peak/1e9, (peak-WEIGHTS)/1e9, capped

# ---- scoring ----
def norm(s): s=s.lower(); s=re.sub(r'\b(a|an|the)\b',' ',s); s="".join(c for c in s if c not in string.punctuation); return " ".join(s.split())
def f1(p,g):
    p=p.strip().split("\n")[0]; pt,gt=norm(p).split(),norm(g).split()
    if not pt or not gt: return float(pt==gt)
    cm=collections.Counter(pt)&collections.Counter(gt); ns=sum(cm.values())
    if ns==0: return 0.0
    pr,rc=ns/len(pt),ns/len(gt); return 2*pr*rc/(pr+rc)
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

# ---- prompt construction ----
def paras(ex):  # list of (title, paragraph_text)
    C=ex["context"]; return [(t," ".join(s)) for t,s in zip(C["title"],C["sentences"])]
if TASK=="gsm8k":
    tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
    DEMOS="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
    INSTR="You are a careful math tutor. Read the problem, reason step by step, and end with the final numeric answer.\n\n"
    pre_ids=tok(INSTR+DEMOS,add_special_tokens=False).input_ids
    it=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))
    for _ in range(580): next(it)
    def build():
        ex=next(it); q=tok(f"Question: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
        return torch.tensor([pre_ids+q],device=dev), lastnum(ex["answer"]), f1  # f1 unused
    scorer=lambda a,g: float(lastnum(a)==g)
else:
    pool_it=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
    POOL=[]  # distractor paragraphs from validation[0:60]
    for _ in range(60):
        for t,p in paras(next(pool_it)): POOL.append(p)
    it=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="validation",streaming=True))
    for _ in range(600): next(it)
    pool_ptr=[0]
    def build():
        ex=next(it); gold_titles=set(ex["supporting_facts"]["title"]); ps=paras(ex)
        chosen=[p for t,p in ps if t in gold_titles]              # gold first, always kept
        chosen+=[p for t,p in ps if t not in gold_titles]         # then example distractors
        ctx=" ".join(chosen); cids=tok(ctx,add_special_tokens=False).input_ids
        while len(cids)<TGT:                                       # pad with pool distractors
            chosen.append(POOL[pool_ptr[0]%len(POOL)]); pool_ptr[0]+=1
            cids=tok(" ".join(chosen),add_special_tokens=False).input_ids
        cids=cids[:TGT]                                            # truncate tail distractor to target
        q=tok(f"\n\nQuestion: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
        return torch.tensor([cids+q],device=dev), ex["answer"], f1
    scorer=lambda a,g: f1(a,g)

conds=["base","native","reuse"]
Q={c:[] for c in conds}; TT={c:[] for c in conds}; LAT={c:[] for c in conds}
PK={c:[] for c in conds}; PKO={c:[] for c in conds}; CAP={c:[] for c in conds}
prefill_once=[]  # one-time shared base-prefix prefill cost (ms), amortized across specialists
for i in range(N):
    full,gold,_=build(); T=full.shape[1]
    sync(); t0=time.perf_counter(); bkv=base_kv(full); sync(); prefill_once.append((time.perf_counter()-t0)*1000)
    for c in conds:
        txt,ttft,total,pk,pko,capped=run(c,full,bkv if c=="reuse" else None)
        Q[c].append(scorer(txt,gold)); TT[c].append(ttft); LAT[c].append(total)
        PK[c].append(pk); PKO[c].append(pko); CAP[c].append(float(capped))
    if (i+1)%50==0: print(f"  {TASK}/{TGT} {i+1}/{N}",flush=True)

A={k:{c:np.array(v[c]) for c in conds} for k,v in [("Q",Q),("TT",TT),("LAT",LAT),("PK",PK),("PKO",PKO),("CAP",CAP)]}
def bci(d,B=5000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
qlab="EM" if TASK=="gsm8k" else "F1"
print(f"\n=== matched: {TASK} target={TGT} tok, n={N} (context≈{full.shape[1]} tok) ===",flush=True)
print(f"one-time shared base-prefix prefill: {np.mean(prefill_once):.0f} ms (amortized across specialists)",flush=True)
print(f"{'cond':>7} {qlab:>6} {'Δ vs native':>20} {'TTFT ms':>9} {'compl ms':>9} {'peak GB':>8} {'over-wt GB':>10} {'cap%':>6}",flush=True)
for c in conds:
    if c=="native": dd=""
    else: m,lo,hi=bci(A["Q"][c]-A["Q"]["native"]); dd=f"{m*100:+.1f} [{lo*100:+.1f},{hi*100:+.1f}]"
    print(f"{c:>7} {A['Q'][c].mean()*100:5.1f} {dd:>20} {A['TT'][c].mean():8.1f} {A['LAT'][c].mean():8.1f} {A['PK'][c].mean():7.2f} {A['PKO'][c].mean():9.2f} {A['CAP'][c].mean()*100:5.1f}",flush=True)
np.savez(f"matched_{TASK}_{TGT}.npz", **{f"{k}_{c}":A[k][c] for k in A for c in conds}, prefill_once=np.array(prefill_once))
print(f"MATCHED_DONE {TASK} {TGT}",flush=True)
