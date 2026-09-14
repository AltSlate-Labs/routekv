"""Generate RouteKV paper figures (SVG) — token-strip schematic, systems savings, boundary result, forest."""
import matplotlib as mpl, matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import Rectangle, Patch
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none","axes.spshort":True} if False else {})
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
BASE="#4C72B0"; SPEC="#DD8452"; GEN="#55A868"; GREY="#9AA0A6"; ACC="#C44E52"; ND="#DDDDDD"
OUT="figures/"

# ---- Fig 1: takeover-boundary token-strip schematic ----
segs=[("instructions",24),("demonstrations",550),("question",82)]
tot=sum(w for _,w in segs); GENW=160
rows=[  # name, n reused-base tokens (boundary), delta label
 ("Native reference",0,"$\\Delta$ —"),
 ("Early takeover",24,"$\\Delta$ +2.2"),
 ("Question takeover",24+550,"$\\Delta$ −3.8"),
 ("Full-prefix reuse",24+550+82,"$\\Delta$ −1.0"),
]
fig,ax=plt.subplots(figsize=(7.4,2.9))
y=0; rh=0.62; gap=0.34
xseg=[0];
for _,w in segs: xseg.append(xseg[-1]+w)
for name,bnd,dl in reversed(rows):
    x=0
    for (lbl,w),x0 in zip(segs,xseg[:-1]):
        col=BASE if x0+w<=bnd else (SPEC if x0<bnd else SPEC)
        # a segment is reused-base if it lies fully within boundary, else specialist-computed
        col=BASE if (x0+w)<=bnd else SPEC
        ax.add_patch(Rectangle((x0,y),w,rh,facecolor=col,edgecolor="white",lw=1.2))
        x=x0+w
    ax.add_patch(Rectangle((tot,y),GENW,rh,facecolor=GEN,edgecolor="white",lw=1.2))
    ax.text(-12,y+rh/2,name,ha="right",va="center",fontsize=9.5)
    ax.text(tot+GENW+14,y+rh/2,dl,ha="left",va="center",fontsize=9.5)
    y+=rh+gap
# segment labels on top
ytop=y-gap+0.06
for (lbl,w),x0 in zip(segs,xseg[:-1]):
    ax.text(x0+w/2,ytop+0.16,f"{lbl}\n({w} tok)",ha="center",va="bottom",fontsize=8.2,color="#333")
ax.text(tot+GENW/2,ytop+0.16,"answer\n(generated)",ha="center",va="bottom",fontsize=8.2,color="#333")
ax.set_xlim(-140,tot+GENW+90); ax.set_ylim(-0.15,ytop+0.7)
ax.axis("off")
leg=[Patch(facecolor=BASE,label="reused base-computed KV"),Patch(facecolor=SPEC,label="specialist-computed"),Patch(facecolor=GEN,label="answer (generated)")]
ax.legend(handles=leg,loc="lower center",bbox_to_anchor=(0.5,-0.13),ncol=3,frameon=False,fontsize=8.4,handlelength=1.1)
plt.tight_layout(); plt.savefig(OUT+"boundary_schematic.svg",bbox_inches="tight"); plt.savefig(OUT+"boundary_schematic.png",bbox_inches="tight",dpi=150); plt.close()

# ---- Fig 2: systems savings (8K ctx) ----
Ms=[2,4]; base_prefill=408; base_cache=0.94  # per-shared-prefix, once
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
x=np.arange(len(Ms)); w=0.36
for ax,(native,reuse,ylab,title,unit) in zip(axes,[
    ([m*base_prefill for m in Ms],[base_prefill]*len(Ms),"prefill latency (ms)","Shared-prefix prefill","ms"),
    ([m*base_cache for m in Ms],[base_cache]*len(Ms),"prefix cache (GB)","Shared-prefix KV cache","GB")]):
    b1=ax.bar(x-w/2,native,w,label="native (per-specialist)",color=GREY)
    b2=ax.bar(x+w/2,reuse,w,label="reuse (once)",color=BASE)
    for i,m in enumerate(Ms):
        ax.text(x[i],max(native)*0.5,f"{m}×",ha="center",va="center",fontsize=11,fontweight="bold",color=ACC)
    ax.set_xticks(x); ax.set_xticklabels([f"M={m}" for m in Ms]); ax.set_ylabel(ylab); ax.set_title(title,fontsize=9.5)
    ax.spines[["top","right"]].set_visible(False)
axes[0].legend(frameon=False,fontsize=8,loc="upper left")
fig.suptitle("Serving M specialists over one 8K shared context (Qwen3-1.7B)",fontsize=9.5,y=1.02)
plt.tight_layout(); plt.savefig(OUT+"savings.svg",bbox_inches="tight"); plt.close()

# ---- Fig 3: boundary result, Delta EM vs reused amount (non-monotonic) ----
conds=["native","early","question","full-prefix"]
reused=[0,24,574,655]; dEM=[0,2.2,-3.8,-1.0]; lo=[0,-0.2,-7.6,-5.0]; hi=[0,4.8,0.0,3.0]
fig,ax=plt.subplots(figsize=(5.4,3.1))
xs=np.arange(len(conds))
yerr=[[d-l for d,l in zip(dEM,lo)],[h-d for d,h in zip(dEM,hi)]]
ax.axhline(0,color="#888",lw=1,ls="--")
cols=[GREY,GEN,ACC,BASE]
ax.errorbar(xs,dEM,yerr=yerr,fmt="none",ecolor="#555",capsize=4,lw=1.3,zorder=1)
ax.scatter(xs,dEM,c=cols,s=70,zorder=3,edgecolor="white",lw=1)
for i,(c,d) in enumerate(zip(conds,dEM)):
    ax.annotate(f"{d:+.1f}" if i else "0.0",(xs[i],dEM[i]),textcoords="offset points",xytext=(0,10 if i!=2 else -16),ha="center",fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels([f"{c}\n({r} tok reused)" for c,r in zip(conds,reused)],fontsize=8.3)
ax.set_ylabel("$\\Delta$ EM vs native (pp)"); ax.set_title("Takeover boundary (GSM8K, math specialist, n=500)",fontsize=9.5)
ax.set_ylim(-11,7); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+"boundary_result.svg",bbox_inches="tight"); plt.savefig(OUT+"boundary_result.png",bbox_inches="tight",dpi=150); plt.close()

# ---- Fig 4: forest plot of specialist-dependence contrasts ----
rows=[  # label, est, lo, hi, n
 ("math reuse penalty (base−native)",-9.2,-17.5,-0.8,120),
 ("QA reuse penalty (base−native)",0.0,-9.2,9.2,120),
 ("penalty difference (math−QA)",-9.2,-21.7,2.5,120),
 ("math seam (question−full-prefix)",-2.8,-6.6,1.0,500),
 ("QA seam (question−full-prefix)",-5.4,-10.6,-0.2,500),
 ("DiD (math seam − QA seam)",2.6,-3.6,8.6,500),
]
fig,ax=plt.subplots(figsize=(6.6,3.3))
ys=np.arange(len(rows))[::-1]
for y,(lbl,e,l,h,n) in zip(ys,rows):
    sig = (l>0 or h<0)
    c=ACC if sig else GREY
    ax.plot([l,h],[y,y],color=c,lw=2.2,solid_capstyle="round")
    ax.scatter([e],[y],color=c,s=55,zorder=3,edgecolor="white",lw=1)
    ax.text(9.6,y,f"n={n}",va="center",fontsize=7.6,color="#666")
ax.axvline(0,color="#333",lw=1)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows],fontsize=8.4)
ax.set_xlabel("effect (pp), 95% paired-bootstrap CI"); ax.set_xlim(-23,12)
ax.set_title("Specialist-dependence contrasts (differences include zero)",fontsize=9.5)
ax.spines[["top","right","left"]].set_visible(False)
leg=[Patch(color=ACC,label="excludes 0"),Patch(color=GREY,label="crosses 0")]
ax.legend(handles=leg,loc="lower left",frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig(OUT+"forest.svg",bbox_inches="tight"); plt.savefig(OUT+"forest.png",bbox_inches="tight",dpi=150); plt.close()
print("wrote 4 figures")
