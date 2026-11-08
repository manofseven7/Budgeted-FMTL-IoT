"""Validate supplements, paired families, and regenerate scientific figures."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/supplement_analysis'

def ci(x): return float(stats.t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x)))
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=json.loads((ROOT/'configs/supplement_executed_plan.json').read_text())
    rows=[];checks=[]
    for conf in plan:
        p=ROOT/conf['output'];m=json.loads((p/'run_manifest.json').read_text())
        df=pd.read_csv(p/'round_metrics.csv');part=pd.read_csv(p/'participation.csv')
        assert len(df)==30 and df['round'].iloc[-1]==300 and df['round'].is_unique
        assert np.isfinite(df.select_dtypes('number')).all().all()
        assert part['rounds'].sum()==7200 and (df.active_clients==24).all()
        assert int(df.cumulative_payload_bytes.iloc[-1])==m['payload_bytes']
        assert (df.cumulative_payload_bytes.diff().dropna()>=0).all()
        assert hashlib.sha256((ROOT/conf['data']).read_bytes()).hexdigest()==m['data_sha256']
        dataset=Path(conf['data']).stem; mix=conf.get('fairness_mix',0)
        row=dict(dataset=dataset,method=m['method'],mix=mix,seed=m['seed'],lr=m['lr'],task1=m['final_metrics'][0],task2=m['final_metrics'][1],
                 payload_mb=m['payload_bytes']/1e6,selection_mb=m['selection_report_bytes']/1e6,jain=m['jain'],wall_seconds=m['wall_seconds'],run=str(p.relative_to(ROOT)))
        assert np.allclose([df.task1_metric.iloc[-1],df.task2_metric.iloc[-1]],m['final_metrics'])
        rows.append(row);checks.append(dict(run=row['run'],status='pass'))
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'seed_level.csv',index=False)
    summaries=[]
    for key,g in raw.groupby(['dataset','method','mix']):
        assert set(g.seed)=={11,22,33,44,55}
        r=dict(zip(['dataset','method','mix'],key));r['n']=len(g)
        for name in ['task1','task2','payload_mb','selection_mb','jain','wall_seconds']:
            r[name+'_mean']=g[name].mean();r[name+'_ci95']=ci(g[name])
        summaries.append(r)
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    legacy=pd.read_csv(ROOT/'results/analysis_extended/extended_seed_level.csv')
    old=legacy[(legacy.family=='main')&legacy.dataset.isin(['uci_har','cwru'])].copy()
    old=old.rename(columns={'jain_participation':'jain'});old['mix']=0;old['selection_mb']=0
    complete=pd.concat([old[['dataset','method','mix','seed','task1','task2','payload_mb','jain','selection_mb']],raw],ignore_index=True)
    complete.to_csv(OUT/'combined_seed_level.csv',index=False)
    comparisons=[]
    for dataset,g in complete[complete['mix']==0].groupby('dataset'):
        proposed=g[g.method=='proposed'].set_index('seed').sort_index()
        family=[]
        for method,h in g[g.method!='proposed'].groupby('method'):
            h=h.set_index('seed').sort_index();assert proposed.index.equals(h.index)
            for outcome in ['task1','task2']:
                delta=proposed[outcome].values-h[outcome].values
                p=1. if np.allclose(delta,0) else float(stats.ttest_1samp(delta,0).pvalue)
                family.append(dict(dataset=dataset,comparator=method,outcome=outcome,difference=float(delta.mean()),ci_low=float(delta.mean()-ci(delta)),ci_high=float(delta.mean()+ci(delta)),p=p))
        order=np.argsort([r['p'] for r in family]);last=0
        for rank,i in enumerate(order):
            last=max(last,min(1.,family[i]['p']*(len(family)-rank)));family[i]['holm_p']=last
        comparisons+=family
    pd.DataFrame(comparisons).to_csv(OUT/'paired_holm.csv',index=False)
    # Fairness endpoints reuse archived proposed results for UCI/CWRU.
    frontier=pd.concat([raw[raw.method=='proposed'],old[old.method=='proposed']],ignore_index=True)
    frontier.to_csv(OUT/'fairness_seed_level.csv',index=False)
    plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.2,'savefig.bbox':'tight'})
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.5))
    for ax,(dataset,g) in zip(axes,frontier.groupby('dataset',sort=True)):
        agg=g.groupby('mix')[['jain','task1']].mean();ax.plot(agg.jain,agg.task1,'o-')
        for mix,r in agg.iterrows():ax.annotate(str(mix),(r.jain,r.task1),fontsize=6)
        ax.set(title=dataset.replace('_',' '),xlabel='Jain participation index',ylabel='Temperature NMSE' if dataset.startswith('intel') else 'Task-1 accuracy')
    fig.tight_layout();fig.savefig(OUT/'fig_fairness_frontier.pdf');plt.close(fig)
    # Inspect saved model metrics independently for Intel finals.
    from src.supplementary import evaluate
    from src.experiment import Layout
    max_error=0
    for row in rows:
        d=np.load(ROOT/f"data/processed/{row['dataset']}.npz")
        types=tuple(d['task_type']) if 'task_type' in d else ('classification','classification')
        layout=Layout(d['x_train'].shape[1],16,tuple(1 if types[t]=='regression' else int(d[f'y{t+1}_train'].max()+1) for t in (0,1)),types)
        s=np.load(ROOT/row['run']/'final_state.npz');personal=np.zeros((len(s['clients']),layout.parameters),dtype='float32')
        if row['method']=='fedrep':personal[:,layout.shared.size:]=s['personal_heads']
        actual=evaluate(s['models'],layout,d,'test',row['method'],personal,s['cluster'],s['clients'])
        max_error=max(max_error,float(np.max(np.abs(np.array(actual)-[row['task1'],row['task2']]))))
    assert max_error<1e-6
    d=np.load(ROOT/'data/processed/intel_timestamp_v2.npz')
    raw_current=d['x_test'][:,:2]*d['x_std'][:2]+d['x_mean'][:2]
    prediction=(raw_current-d['y_mean'])/d['y_std']
    actual=np.stack([d['y1_test'],d['y2_test']],axis=1)
    mse=np.mean((prediction-actual)**2,axis=0)
    persistence=dict(method='last_observation_persistence',test_rows=len(actual),normalized_mse=mse.tolist(),
                     physical_rmse=(np.sqrt(mse)*d['y_std']).tolist(),training_required=False,seed_dependent=False,
                     payload_bytes=0,notes='Local task-specific sanity reference; not a communication-matched FL comparator.')
    (OUT/'persistence.json').write_text(json.dumps(persistence,indent=2))
    assert d['target_timestamp_train'].max()<d['timestamp_val'].min()
    assert d['target_timestamp_val'].max()<d['timestamp_test'].min()
    assert set(d['client_test']).issubset(set(d['client_train']))
    audit=dict(complete_new_runs=len(rows),tuning_runs=18,planned_runs=len(plan),all_valid=True,full_rounds_per_run=300,
               logged_evaluation_checkpoints_per_run=30,training_rounds=sum(c['rounds'] for c in plan)+18*60,
               independent_final_metric_max_error=max_error,holm_family='all comparator x 2 task tests within each dataset; fairness sweeps diagnostic',
               archived_runs_retrained=0,intel_legacy_policy='Excluded from current claims; timestamp-v2 is a new corrected protocol')
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
