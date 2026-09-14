"""RouteKV step 2 (augmented): boundary confirm + specialist-dependence DiD, n>=500 math examples.
Per example, computes ALL outcomes together (pairing guaranteed by construction):
  math adapter: native, early, question, full_prefix   (boundary confirm + controls)
  qa   adapter: question, full_prefix                   (for the DiD)
Shardable: argv = START COUNT (over test examples after skipping the 80-example dev set). Writes per-example
0/1 outcome arrays to a .npz shard; merge_did.py concatenates shards and does all bootstraps.
DiD estimand: (question - full_prefix)_math - (question - full_prefix)_qa, paired over the 4 aligned outcomes.
"""
import sys, torch, re, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from peft import PeftModel
from datasets import load_dataset

START=int(sys.argv[1]); COUNT=int(sys.argv[2])
BACKBONE="Qwen/Qwen3-1.7B"; NDEMO=4; MAXNEW=160; DEVSKIP=80; dev="cuda"
tok=AutoTokenizer.from_pretrained(BACKBONE)
base=AutoModelForCausalLM.from_pretrained(BACKBONE,dtype=torch.bfloat16).cuda().eval()
model=PeftModel.from_pretrained(base,"adapters/math",adapter_name="math"); model.load_adapter("adapters/qa",adapter_name="qa"); model.eval()
NL=base.config.num_hidden_layers

INSTR="You are a careful math tutor. Read the problem, reason step by step, and end with the final numeric answer.\n\n"
tr=iter(load_dataset("openai/gsm8k","main",split="train",streaming=True))
DEMOS="".join(f"Question: {e['question']}\nAnswer: {e['answer']}\n\n" for e in [next(tr) for _ in range(NDEMO)])
instr_ids=tok(INSTR,add_special_tokens=False).input_ids; demo_ids=tok(DEMOS,add_special_tokens=False).input_ids
L_INSTR=len(instr_ids); L_DEMO=len(demo_ids)

def base_kv(ids):
    with model.disable_adapter(),torch.inference_mode(): pkv=model(input_ids=ids,use_cache=True).past_key_values
    return [(pkv.layers[l].keys, pkv.layers[l].values) for l in range(NL)]
def cache_slice(bkv,pb):
    c=DynamicCache()
    for l,(k,v) in enumerate(bkv): c.update(k[:,:,:pb,:].clone(),v[:,:,:pb,:].clone(),l)
    return c
def greedy(adapter,cache,primer):
    model.set_adapter(adapter)
    with torch.inference_mode(): out=model(input_ids=primer,past_key_values=cache,use_cache=True)
    lg=out.logits[0,-1]; cache=out.past_key_values; toks=[]
    for _ in range(MAXNEW):
        nt=int(lg.argmax())
        if nt==tok.eos_token_id: break
        toks.append(nt)
        with torch.inference_mode(): out=model(input_ids=torch.tensor([[nt]],device=dev),past_key_values=cache,use_cache=True); lg=out.logits[0,-1]; cache=out.past_key_values
    return tok.decode(toks)
def lastnum(s): m=re.findall(r'-?\d[\d,]*',s.replace(",","")); return m[-1] if m else ""

keys=["math_native","math_early","math_question","math_full","qa_question","qa_full"]
out={k:[] for k in keys}; golds=[]
te=iter(load_dataset("openai/gsm8k","main",split="test",streaming=True))
for _ in range(DEVSKIP+START): next(te)
for i in range(COUNT):
    ex=next(te); gold=lastnum(ex["answer"]); golds.append(gold)
    q_ids=tok(f"Question: {ex['question']}\nAnswer:",add_special_tokens=False).input_ids
    full=torch.tensor([instr_ids+demo_ids+q_ids],device=dev); T=full.shape[1]
    bkv=base_kv(full)
    b_q=L_INSTR+L_DEMO; b_fp=T-1  # question boundary, full-prefix boundary
    sc=lambda a:float(lastnum(a)==gold)
    out["math_native"].append(sc(greedy("math",DynamicCache(),full)))
    out["math_early"].append(sc(greedy("math",cache_slice(bkv,L_INSTR),full[:,L_INSTR:])))
    out["math_question"].append(sc(greedy("math",cache_slice(bkv,b_q),full[:,b_q:])))
    out["math_full"].append(sc(greedy("math",cache_slice(bkv,b_fp),full[:,b_fp:])))
    out["qa_question"].append(sc(greedy("qa",cache_slice(bkv,b_q),full[:,b_q:])))
    out["qa_full"].append(sc(greedy("qa",cache_slice(bkv,b_fp),full[:,b_fp:])))
    if (i+1)%50==0: print(f"  shard[{START}:{START+COUNT}] {i+1}/{COUNT}",flush=True)

np.savez(f"did_shard_{START}_{COUNT}.npz", golds=np.array(golds), **{k:np.array(v) for k,v in out.items()})
print(f"SHARD_DONE {START} {COUNT}",flush=True)
