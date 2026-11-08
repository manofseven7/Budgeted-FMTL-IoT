"""Generate figures from audited runs; excludes legacy Intel from current claims."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.analyze_extended import ci95, telemetry_figure
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/supplement_analysis/figures'
OUT.mkdir(parents=True,exist_ok=True)
old=pd.read_csv(ROOT/'results/analysis_extended/extended_seed_level.csv');old=old[old.dataset!='intel_sensor']
data=pd.read_csv(ROOT/'results/supplement_analysis/combined_seed_level.csv');data=data[data['mix']==0]
datasets=['uci_har','cwru','intel_timestamp_v2']
names={'fedavg':'FedAvg','fedprox':'FedProx','fmtl':'Laplacian FMTL','scaffold':'SCAFFOLD','topksgd':'TopK-SGD','proposed':'Proposed','poc':'Power-of-Choice','fedrep':'FedRep adaptation','ifca':'IFCA adaptation'}
plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.2,'savefig.bbox':'tight'})
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5))
for ax,ds in zip(axes,datasets):
    g=data[data.dataset==ds];methods=list(g.method.unique())
    ax.bar(range(len(methods)),[g[g.method==m].task1.mean() for m in methods],yerr=[ci95(g[g.method==m].task1) for m in methods],capsize=2)
    ax.set_xticks(range(len(methods)),[names[m].replace(' adaptation','') for m in methods],rotation=60,ha='right',fontsize=6)
    ax.set(title=ds.replace('_',' '),ylabel='Temperature NMSE' if ds.startswith('intel') else 'Task-1 accuracy')
fig.tight_layout();fig.savefig(OUT/'fig_task1_performance.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5))
for ax,ds in zip(axes,datasets):
    for method,g in data[data.dataset==ds].groupby('method'):
        curves=[]
        for _,r in g.iterrows():
            if isinstance(r.get('run'),str):path=ROOT/r['run']/'round_metrics.csv'
            else:
                path=ROOT/('results/v2/main' if method=='proposed' else 'results/full')/f'{ds}_{method}_{int(r.seed)}'/'round_metrics.csv'
            curves.append(pd.read_csv(path).set_index('round').task1_metric)
        frame=pd.concat(curves,axis=1);mean=frame.mean(axis=1);sd=frame.std(axis=1)
        ax.plot(mean.index,mean,label=names[method],linewidth=.9);ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.08)
    ax.set(title=ds.replace('_',' '),xlabel='Round',ylabel='NMSE' if ds.startswith('intel') else 'Accuracy')
axes[1].legend(fontsize=5,ncol=2);fig.tight_layout();fig.savefig(OUT/'fig_convergence.pdf');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(7.2,2.5))
for ax,ds in zip(axes,datasets):
    g=data[data.dataset==ds].groupby('method').payload_mb.mean()
    ax.bar(range(len(g)),g);ax.set_xticks(range(len(g)),[names[m].replace(' adaptation','') for m in g.index],rotation=60,ha='right',fontsize=6)
    ax.set(title=ds.replace('_',' '),ylabel='Model-update uplink (MB)')
fig.tight_layout();fig.savefig(OUT/'fig_payload.pdf');plt.close(fig)
variants=['full','no_scheduler','no_taskaware','no_ef','no_freeze']
fig,axes=plt.subplots(1,2,figsize=(7.2,2.5))
for ax,ds in zip(axes,datasets[:2]):
    q=old[(old.dataset==ds)&(old.family=='component')]
    ax.bar(range(5),[q[q.variant==v].task1.mean() for v in variants],yerr=[ci95(q[q.variant==v].task1) for v in variants],capsize=2)
    ax.set_xticks(range(5),['Full','No scheduler','No overlap','No EF','No freezing'],rotation=25,ha='right');ax.set(title=ds.replace('_',' '),ylabel='Accuracy')
fig.tight_layout();fig.savefig(OUT/'fig_component_ablation.pdf');plt.close(fig)
for family,x,xlabel in [('rho','rho','Retention'),('sigma','sigma','Similarity threshold'),('eta','eta_f','Freezing threshold')]:
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.3))
    for ax,ds in zip(axes,datasets[:2]):
        q=old[(old.dataset==ds)&(old.family==family)]
        means=q.groupby(x).task1.mean();errors=q.groupby(x).task1.apply(ci95)
        ax.errorbar(means.index,means,yerr=errors,marker='o',capsize=2)
        ax.set(title=ds.replace('_',' '),xlabel=xlabel,ylabel='Task-1 accuracy')
    fig.tight_layout();fig.savefig(OUT/f'fig_{family}_sensitivity.pdf');plt.close(fig)
telemetry_figure(old,OUT/'fig_sigma_telemetry.pdf')
fig,axes=plt.subplots(1,2,figsize=(7.2,2.4))
for ds,g in old[old.family=='component'].groupby('dataset'):
    q=g[g.variant.isin(['full','no_scheduler'])];means=q.groupby('variant').jain_participation.mean()
    for v,r in means.items():axes[0].scatter(ds+'\n'+v,r)
    full=g[g.variant=='full'];axes[1].scatter(full.mean_budget_utilization,full.jain_participation,label=ds)
axes[0].set_ylabel('Jain participation index');axes[0].tick_params(axis='x',labelsize=6)
axes[1].set(xlabel='Synthetic budget utilization',ylabel='Jain participation index');axes[1].legend(fontsize=7)
fig.tight_layout();fig.savefig(OUT/'fig_participation_budget.pdf');plt.close(fig)
print('Generated figures:',len(list(OUT.glob('fig_*.pdf'))))
