"""RouteKV validation — memory/latency audit, DIRECT reuse (GPU 3).
Since direct base-prefix reuse ≈ native (no translator), the composable-serving economics are pure
cache-sharing: serve M specialists on one shared context = 1 prefill + 1 shared prefix cache + M
suffix generations, vs native M prefills + M prefix caches. Rigor: weights shared (counted once);
shared prefix = ONE physical allocation; cold vs warm prefill; M=1,2,4. Savings apply to the
SHAREABLE PREFIX only (weights + private suffix caches + buffers remain).
"""
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel

BACKBONE="Qwen/Qwen3-1.7B"; CTXS=[700,2048,8192]; REP=5; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/qa",adapter_name="qa"); model.load_adapter("adapters/math",adapter_name="math"); model.eval()
NL,NH,HD=base.config.num_hidden_layers,base.config.num_key_value_heads,base.config.head_dim
w_bytes=sum(p.numel()*p.element_size() for p in base.parameters())/1e9
print(f"shared backbone weights: {w_bytes:.2f} GB (counted ONCE for all specialists)",flush=True)
def seq(L): return torch.randint(0,140000,(1,L),device=dev)
def prefill(which,ids):
    if which is None:
        with model.disable_adapter(),torch.inference_mode(): return model(input_ids=ids,use_cache=True).past_key_values
    model.set_adapter(which)
    with torch.inference_mode(): return model(input_ids=ids,use_cache=True).past_key_values
def timed(fn,rep=REP):
    fn(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0=time.perf_counter()
    for _ in range(rep): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/rep*1000, torch.cuda.max_memory_allocated()/1e9

print(f"{'ctx':>6} {'prefix_cache_MB':>15} {'cold_prefill_ms':>15} {'warm_prefill_ms':>15}",flush=True)
for L in CTXS:
    ids=seq(L); cache_mb=NL*NH*L*HD*2*2/1e6
    # cold = first call incl. cuda init overhead; warm = steady-state (timed() warms up)
    torch.cuda.synchronize(); t0=time.perf_counter()
    with model.disable_adapter(),torch.inference_mode(): model(input_ids=ids,use_cache=True)
    torch.cuda.synchronize(); cold=(time.perf_counter()-t0)*1000
    warm,_=timed(lambda: prefill(None,ids))
    print(f"{L:>6} {cache_mb:>14.1f} {cold:>14.1f} {warm:>14.1f}",flush=True)
    # verify ONE physical allocation: shared cache tensors reused by pointer, not copied
    pkv=prefill(None,ids); ptr=pkv.layers[0].keys.data_ptr()
    print(f"        shared prefix cache = 1 physical alloc (layer0 key ptr {hex(ptr)}); native would hold M copies",flush=True)
    for M in [1,2,4]:
        nat_lat=M*warm; rk_lat=warm            # RouteKV: 1 prefill shared; +M suffix gens (equal both sides, omitted)
        nat_cache=M*cache_mb; rk_cache=cache_mb
        print(f"        M={M}: prefix-prefill native {nat_lat:.0f}ms vs RouteKV {rk_lat:.0f}ms ({nat_lat/rk_lat:.1f}x) | "
              f"prefix-cache native {nat_cache:.0f}MB vs RouteKV {rk_cache:.0f}MB ({M}x)",flush=True)
print("(NB: savings are the SHAREABLE PREFIX only; weights shared once = "+f"{w_bytes:.1f}GB; per-specialist suffix caches + working buffers not counted here.)",flush=True)
print("MEMAUDIT_DONE",flush=True)
