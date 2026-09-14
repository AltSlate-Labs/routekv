"""Merge confirm_did.py shards -> boundary-confirm table + specialist-dependence DiD."""
import glob, numpy as np
keys=["math_native","math_early","math_question","math_full","qa_question","qa_full"]
shards=sorted(glob.glob("did_shard_*.npz"), key=lambda p:int(p.split("_")[2]))
D={k:np.concatenate([np.load(s)[k] for s in shards]) for k in keys}
N=len(D["math_native"]); print(f"merged {len(shards)} shards, n={N}",flush=True)
def bci(d,B=10000): idx=np.random.randint(0,len(d),(B,len(d))); m=d[idx].mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)

print(f"\n=== Step 2 boundary confirm (math, n={N}) ===",flush=True)
nat=D["math_native"]
print(f"native EM={nat.mean()*100:.1f}%",flush=True)
for c,arr in [("early",D["math_early"]),("question",D["math_question"]),("full_prefix",D["math_full"])]:
    m,lo,hi=bci(arr-nat); print(f"  {c:>11}: EM {arr.mean()*100:5.1f}  Δ vs native {m*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
m,lo,hi=bci(D["math_question"]-D["math_full"])
print(f"\nDirect seam contrast (math)  question - full_prefix : {m*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]  (seam<0)",flush=True)

print(f"\n=== Specialist-dependence DiD (n={N}, paired over 4 outcomes) ===",flush=True)
seam_math=D["math_question"]-D["math_full"]; seam_qa=D["qa_question"]-D["qa_full"]
for nm,d in [("math adapter seam (q - fp)",seam_math),("qa adapter seam (q - fp)",seam_qa)]:
    m,lo,hi=bci(d); print(f"  {nm:>28}: {m*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
m,lo,hi=bci(seam_math-seam_qa)
print(f"  {'DiD (math seam - qa seam)':>28}: {m*100:+.1f}pp [{lo*100:+.1f},{hi*100:+.1f}]",flush=True)
print(f"\n  (qa question EM {D['qa_question'].mean()*100:.1f}, qa full_prefix EM {D['qa_full'].mean()*100:.1f})",flush=True)
print("MERGE_DONE",flush=True)
