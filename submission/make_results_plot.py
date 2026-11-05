"""Render a scientific summary figure from measured data; no API."""
from pathlib import Path
import json,os
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=json.loads((ROOT/'reports/cycle_stability_20261104.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(9.5,3.05),layout='constrained')
xs=[1,2,3]
cc=[r['arms']['cycle_control_3']['calls'] for r in s['rounds']];pc=[r['arms']['cycle_pruned_3']['calls'] for r in s['rounds']]
cs=[100*r['arms']['cycle_control_3']['success_rate'] for r in s['rounds']];ps=[100*r['arms']['cycle_pruned_3']['success_rate'] for r in s['rounds']]
for ax,y,z,title in [(axes[0],cc,pc,'Completed API calls'),(axes[1],cs,ps,'Independent success (%)')]:
 ax.plot(xs,y,'o-',color='#15324b',label='Engineering control',lw=2)
 ax.plot(xs,z,'s-',color='#087e8b',label='Cycle pruning',lw=2)
 ax.set_xticks(xs);ax.set_xlabel('Round (same 20 tasks)');ax.set_title(title,fontsize=11)
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
axes[0].set_ylim(0,1300);axes[1].set_ylim(0,60);axes[0].legend(fontsize=8,loc='lower left');axes[1].set_yticks([0,20,40,60])
(ROOT/'tmp/pdfs').mkdir(parents=True,exist_ok=True)
fig.savefig(ROOT/'tmp/pdfs/main_results.png',dpi=180);plt.close(fig)
