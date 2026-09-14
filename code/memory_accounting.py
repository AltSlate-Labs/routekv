"""RouteKV step 3 — explicit memory + latency accounting for same-base composable serving.

Scenario: serve M specialists on the SAME context (shared backbone, both adapters resident, 1 GPU).
  native serving: each specialist prefills the context itself -> M context prefills + M context caches.
  RouteKV:        1 context prefill (base) + 1 shared context cache + M cheap near-identity translates.
Measures: context cache bytes; specialist native-prefill latency & peak mem; handoff (base prefill +
translate) latency, translate-only latency, translate peak temp; and the M-specialist amortization.
"""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

BACKBONE = "Qwen/Qwen3-1.7B"
N_CAL, LAM = 30, 10.0
CTXS = [700, 2048, 8192]; REP = 5
dev = "cuda"
tok = AutoTokenizer.from_pretrained(BACKBONE)
base = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.bfloat16).cuda().eval()
model = PeftModel.from_pretrained(base, "adapters/qa", adapter_name="qa")
model.load_adapter("adapters/math", adapter_name="math"); model.eval()
NL, NH, HD = base.config.num_hidden_layers, base.config.num_key_value_heads, base.config.head_dim

def rot(x): h=x.shape[-1]//2; return torch.cat([-x[...,h:],x[...,:h]],-1)
def rope(nn):
    p=torch.arange(nn,device=dev).unsqueeze(0); hid=torch.zeros(1,nn,base.config.hidden_size,device=dev,dtype=torch.bfloat16)
    c,s=base.model.rotary_emb(hid,p); return c.unsqueeze(1).float(), s.unsqueeze(1).float()
def strip(k,c,s): return k*c-rot(k)*s
def app(k,c,s): return k*c+rot(k)*s
def kv(which, ids):
    if which is None:
        with model.disable_adapter(), torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    else:
        model.set_adapter(which)
        with torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return (torch.stack([pkv.layers[l].keys[0] for l in range(NL)]).float(),
            torch.stack([pkv.layers[l].values[0] for l in range(NL)]).float())
def build_cache(K,V):
    c=DynamicCache()
    for l in range(NL): c.update(K[l].unsqueeze(0).to(torch.bfloat16),V[l].unsqueeze(0).to(torch.bfloat16),l)
    return c

# ---- fit base->QA near-identity l->l ridge (cheap translator) ----
tr=iter(load_dataset("hotpotqa/hotpot_qa","distractor",split="train",streaming=True))
Sxx={m:torch.zeros(NL,NH,HD,HD,device=dev) for m in "KV"}; Sxy={m:torch.zeros(NL,NH,HD,HD,device=dev) for m in "KV"}
sxa={m:torch.zeros(NL,NH,HD,device=dev) for m in "KV"}; sya={m:torch.zeros(NL,NH,HD,device=dev) for m in "KV"}; n=0
for i in range(N_CAL):
    ex=next(tr); passages="".join(" ".join(s) for s in ex["context"]["sentences"])
    ids=tok(passages,return_tensors="pt").input_ids[:,:700].cuda(); P=ids.shape[1]; cs,sn=rope(P)
    bK,bV=kv(None,ids); qK,qV=kv("qa",ids); bK=strip(bK,cs,sn); qK=strip(qK,cs,sn)
    for m,X,Y in [("K",bK,qK),("V",bV,qV)]:
        Sxx[m]+=torch.einsum('lhsd,lhse->lhde',X,X); Sxy[m]+=torch.einsum('lhsd,lhse->lhde',X,Y)
        sxa[m]+=X.sum(2); sya[m]+=Y.sum(2)
    n+=P
W={};b={}
for m in "KV":
    Cxx=Sxx[m]-torch.einsum('lhd,lhe->lhde',sxa[m],sxa[m])/n; Cxy=Sxy[m]-torch.einsum('lhd,lhe->lhde',sxa[m],sya[m])/n
    W[m]=torch.linalg.solve(Cxx+LAM*torch.eye(HD,device=dev),Cxy); b[m]=sya[m]/n-torch.einsum('lhde,lhd->lhe',W[m],sxa[m]/n)
def translate(bK,bV,P):
    cs,sn=rope(P); Ksf=strip(bK,cs,sn)
    Kh=torch.einsum('lhde,lhsd->lhse',W["K"],Ksf)+b["K"].unsqueeze(2); Vh=torch.einsum('lhde,lhsd->lhse',W["V"],bV)+b["V"].unsqueeze(2)
    return build_cache(app(Kh,cs,sn),Vh)
print("translator fitted",flush=True)

def timed(fn,rep=REP):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/rep*1000, torch.cuda.max_memory_allocated()/1e9

def cal(L):
    b=[]
    while len(b)<L: b+=tok(next(tr)["text"] if False else "the quick brown fox jumps over the lazy dog. ").input_ids
    return torch.tensor(b[:L],device=dev).unsqueeze(0)

print("\n=== step 3: memory + latency accounting (same-base composable serving) ===",flush=True)
print(f"context-cache bytes = NL*NH*P*HD*2(K,V)*2(bf16). backbone weights shared; adapters ~25MB each.",flush=True)
print(f"{'ctx':>6} {'cache_MB':>9} {'QA_prefill_ms':>14} {'base_prefill_ms':>15} {'translate_ms':>13} {'translate_peakGB':>16}",flush=True)
for L in CTXS:
    ids=cal(L); cache_mb = NL*NH*L*HD*2*2/1e6
    def f_qa():
        model.set_adapter("qa")
        with torch.inference_mode(): model(input_ids=ids,use_cache=True)
    def f_base():
        with model.disable_adapter(), torch.inference_mode(): model(input_ids=ids,use_cache=True)
    qa_ms,_=timed(f_qa); bs_ms,_=timed(f_base)
    bK,bV=kv(None,ids)
    tr_ms,tr_pk=timed(lambda: translate(bK,bV,L))
    print(f"{L:>6} {cache_mb:>8.1f} {qa_ms:>13.1f} {bs_ms:>14.1f} {tr_ms:>12.1f} {tr_pk:>15.1f}",flush=True)
    # amortization for M specialists on this context
    for M in [2,4]:
        nat = M*qa_ms; rk = bs_ms + M*tr_ms
        nat_cache = M*cache_mb; rk_cache = cache_mb  # shared
        print(f"        M={M}: latency native {nat:.0f}ms vs RouteKV {rk:.0f}ms ({nat/rk:.1f}x) | "
              f"cache native {nat_cache:.0f}MB vs RouteKV {rk_cache:.0f}MB ({M}x less)",flush=True)
print("MEMACCT_DONE",flush=True)
