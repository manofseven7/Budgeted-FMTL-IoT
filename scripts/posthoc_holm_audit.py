from pathlib import Path
import shutil, pandas as pd, numpy as np
from scipy.stats import ttest_rel, t
root=Path(__file__).resolve().parents[1]; out=root/'results/posthoc'; out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(root/'results/analysis_extended/extended_seed_level.csv'); main=df[df.family=='main'].copy()
rows=[]
for dataset in sorted(main.dataset.unique()):
 for comparator in ['topksgd','fedavg','scaffold','fedprox','fmtl']:
  p=main[(main.dataset==dataset)&(main.method=='proposed')].set_index('seed'); b=main[(main.dataset==dataset)&(main.method==comparator)].set_index('seed'); seeds=sorted(set(p.index)&set(b.index));
  if len(seeds)<2:continue
  for metric in ['task1','task2','payload_mb','jain_participation']:
   d=(p.loc[seeds,metric]-b.loc[seeds,metric]).to_numpy(float); stat,pval=ttest_rel(p.loc[seeds,metric],b.loc[seeds,metric]);
   ci=t.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d));rows.append(dict(dataset=dataset,comparator=comparator,metric=metric,n=len(d),mean_difference=float(d.mean()),ci95_low=float(d.mean()-ci),ci95_high=float(d.mean()+ci),p_raw=float(pval) if np.isfinite(pval) else 1.0))
r=pd.DataFrame(rows); r['p_holm_family']='';
# family: each dataset, metric; five comparator hypotheses
for (dataset,metric),ix in r.groupby(['dataset','metric']).groups.items():
 ix=list(ix); order=sorted(ix,key=lambda i:r.loc[i,'p_raw']); running=0
 for rank,i in enumerate(order):
  running=max(running,min(1.,r.loc[i,'p_raw']*(len(order)-rank)));r.loc[i,'p_holm_family']=running
r.to_csv(out/'paired_comparisons_holm.csv',index=False)
summary=main.groupby(['dataset','method'])[['task1','task2','payload_mb','jain_participation']].agg(['mean','std']);summary.to_csv(out/'main_summary.csv')
# targeted findings
with open(out/'AUDIT_FINDINGS.md','w') as f:
 f.write('# Independent post-hoc audit of archived experimental outputs\n\nThis is a new statistical analysis of archived five-seed runs, **not new training runs**. Paired t tests use the same seed IDs and Holm correction within each dataset × outcome family across five comparators. No causal inference is claimed.\n\n')
 for dataset in sorted(main.dataset.unique()):
  x=r[(r.dataset==dataset)&(r.comparator=='topksgd')]; f.write('## '+dataset+' — proposed minus matched TopK-SGD\n\n');f.write('| Outcome | Difference | 95% CI | Holm p |\n|---|---:|---:|---:|\n')
  for _,a in x.iterrows():f.write(f"| {a.metric} | {a.mean_difference:.6g} | [{a.ci95_low:.6g}, {a.ci95_high:.6g}] | {float(a.p_holm_family):.4g} |\n")
  f.write('\n')
 f.write('## Important source-code and evidence limitations\n\n- The processed UCI HAR, CWRU, and Intel `.npz` datasets are not included in the repository, so no new dataset-based training was run in this audit.\n- `remaining_energy` is a synthetic state that decreases every participation and is clipped to zero. It is not physical energy or a hard per-client feasibility constraint; the scheduler uses a round budget based on synthetic costs.\n- `costs` become identical across clients while freezing is inactive; budget is then 1.05 × the cost of the target number of clients. This makes the round-budget constraint nonbinding at the default setting.\n- `scores` contain a normalized residual-energy term and gradient norm affinity, but the third communication-frugality component is zero for every client when freezing is inactive. Thus no default client-level communication-cost differentiation is demonstrated.\n- The current `fmtl` is a surrogate head coupling, not a faithful reproduction of a named external FMTL solver.\n- Statistical comparisons are exploratory post-hoc; Holm correction mitigates only the declared comparator family, not all possible analysis choices.\n- No new personalized FL, clustered FL, or distinct-task dataset baseline was run; those reviewer concerns remain open.\n')
print(out)
print(r[(r.comparator=='topksgd')&(r.metric=='task1')][['dataset','mean_difference','p_raw','p_holm_family']].to_string(index=False))
