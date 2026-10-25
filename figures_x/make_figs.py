import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base="./results/"
ce=json.load(open(base+"seed_package_CE_gb300.json"))
r=ce["runs"]
out="figures_x/"
rev="figures_x/"
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})

# Figure 1: five arms, two panels (QA avg, BC+), per-seed dots
arms=[("A\nGRPO G=8","A",[0]),("B\nTreePO\n(paper's control)","B",[0,1,2]),
      ("C\nTreeOPSD","C",[0,1,2]),("D\nGRPO G=12\n(token-matched)","D",[0]),
      ("E\nTreePO + clip\n(our control)","E",[0,1,2])]
fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=False)
for ax,key,title in [(axes[0],"qa_avg","Tier 1: multi-hop QA average (EM)"),
                     (axes[1],"bcplus","Tier 2: BC+ long-horizon (EM, 830 Q)")]:
    means=[]; 
    for i,(label,arm,seeds) in enumerate(arms):
        vals=[r[f"{arm}s{s}"][key] for s in seeds]
        means.append(np.mean(vals))
        ax.bar(i,np.mean(vals),width=0.55,color="#b8c7d9",edgecolor="#4a5a6a",zorder=1)
        ax.plot([i]*len(vals),vals,"o",color="#c0392b",ms=6,zorder=3)
        ax.text(i,np.mean(vals)+0.7,f"{np.mean(vals):.1f}",ha="center",fontsize=10)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([a[0] for a in arms],fontsize=8.5)
    ax.set_title(title,fontsize=11); ax.grid(axis="y",alpha=0.25)
    ax.set_ylim(0,50 if key=="qa_avg" else 16)
axes[0].annotate("B seed 2: destabilising\nin last 10 steps",xy=(1,r["Bs2"]["qa_avg"]),
                 xytext=(0.2,18),fontsize=8.5,color="#c0392b",
                 arrowprops=dict(arrowstyle="->",color="#c0392b",lw=1))
axes[0].set_ylabel("exact match")
fig.suptitle("TreeOPSD replication (Qwen3-4B, step 100): bars = seed mean, dots = seeds 0-2",fontsize=12)
fig.tight_layout()
for d in [out,rev]: fig.savefig(d+"fig1-five-arms.png",dpi=200)
print("fig1 saved")

# Figure 2: forest plot of gaps, both intervals
rows=[
 ("C-E  QA  (posterior vs clipped)", 1.98,1.03,2.94, -2.66,6.61),
 ("C-E  BC+ (posterior vs clipped)", 0.08,-1.24,1.41, -10.74,10.90),
 ("C-B  QA  (vs paper's control)", 3.76,2.81,4.74, -4.06,11.57),
 ("C-B  BC+ (vs paper's control)", 3.45,2.13,4.78, -6.13,13.03),
]
fig,ax=plt.subplots(figsize=(10,4.4))
ys=np.arange(len(rows))[::-1]
for y,(label,pt,lo,hi,slo,shi) in zip(ys,rows):
    ax.plot([slo,shi],[y+0.14]*2,color="#95a5a6",lw=5,solid_capstyle="butt")
    ax.plot([lo,hi],[y-0.14]*2,color="#2471a3",lw=5,solid_capstyle="butt")
    ax.plot(pt,y,"o",color="black",ms=5,zorder=5)
    ax.text(14.2,y,f"{pt:+.2f}",va="center",fontsize=10)
ax.axvline(0,color="black",lw=1)
ax.axvline(2.90,color="#c0392b",lw=1,ls="--")
ax.text(2.90,4.35,"paper's claimed BC+ gap +2.90",color="#c0392b",fontsize=9,ha="center")
ax.set_yticks(ys); ax.set_yticklabels([x[0] for x in rows],fontsize=9.5)
ax.set_xlabel("exact-match gap (points)   blue = question bootstrap CI,  grey = seed interval (n=3)")
ax.set_title("The deciding gaps: pooled point (dot) with both uncertainties",fontsize=11.5)
ax.set_xlim(-12,16); ax.grid(axis="x",alpha=0.25)
fig.tight_layout()
for d in [out,rev]: fig.savefig(d+"fig2-gaps.png",dpi=200)
print("fig2 saved")
