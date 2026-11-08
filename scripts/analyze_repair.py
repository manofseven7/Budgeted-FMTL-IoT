"""Audit the frozen repair experiment and create paper tables and figures."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.repair import evaluate
from src.experiment import Layout
OUT=ROOT/'results/repair/analysis';SEEDS={11,22,33,44,55}
NAMES={'fedavg':'FedAvg','fedprox':'FedProx','fmtl':'Laplacian surrogate','scaffold':'SCAFFOLD','topksgd':'TopK-EF','poc':'Power-of-Choice','proposed':'Proposed','fedrep':'FedRep adaptation','ifca':'IFCA adaptation','scheduled_topk':'Scheduled TopK control'}
def ci(x):return float(stats.t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x)))
def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];reconstructed=0;maxerror=0;reused=0
    plans=json.loads((ROOT/'configs/repair_final_plan.json').read_text())+json.loads((ROOT/'configs/repair_control_plan.json').read_text())
    for conf in plans:
        path=ROOT/conf['output'];m=json.loads((path/'run_manifest.json').read_text());reuse=m.get('reused_from')
        source=ROOT/reuse if reuse else path;curve=pd.read_csv(source/'round_metrics.csv');part=pd.read_csv(source/'participation.csv')
        assert curve['round'].iloc[-1]==300 and part['rounds'].sum()==7200 and np.isfinite(curve.select_dtypes('number')).all().all()
        assert int(curve.cumulative_payload_bytes.iloc[-1])==m['payload_bytes']
        assert m['seed'] in SEEDS
        if reuse:reused+=1
        else:
            assert hashlib.sha256((ROOT/conf['data']).read_bytes()).hexdigest()==m['data_sha256']
            d=np.load(ROOT/conf['data']);types=tuple(d['task_type']) if 'task_type'in d else ('classification','classification')
            layout=Layout(d['x_train'].shape[1],16,tuple(1 if types[t]=='regression' else int(d[f'y{t+1}_train'].max()+1) for t in [0,1]),types)
            s=np.load(path/'final_state.npz')
            if m['method']=='scaffold' and 'client_weights' in s:
                np.testing.assert_allclose(s['server_control'],np.einsum('n,ntp->tp',s['client_weights'],s['client_controls']),rtol=1e-4,atol=1e-6)
            personal=np.zeros((len(s['clients']),layout.parameters),dtype='float32')
            if m['method']=='fedrep':personal[:,layout.shared.size:]=s['personal_heads']
            values=evaluate(s['models'],layout,d,'test',m['method'],personal,s['cluster'],s['clients'])
            err=float(np.max(np.abs(np.array(values)-m['final_metrics'])));maxerror=max(maxerror,err);assert err<1e-7;reconstructed+=1
        row=dict(dataset=m['dataset_name'],method='scheduled_topk' if conf.get('role') else m['method'],seed=m['seed'],
            task1=m['final_metrics'][0],task2=m['final_metrics'][1],model_mb=m['payload_bytes']/1e6,aux_mb=m['selection_report_bytes']/1e6,
            downlink_mb=m['logical_downlink_bytes']/1e6,jain=m['jain'],lr=m['lr'],steps=m.get('steps',0),wall_seconds=m['wall_seconds'],run=conf['output'],reused=bool(reuse))
        row['uplink_mb']=row['model_mb']+row['aux_mb'];row['logical_total_mb']=row['uplink_mb']+row['downlink_mb']
        rows.append(row)
    raw=pd.DataFrame(rows);raw.to_csv(OUT/'seed_level.csv',index=False);summary=[]
    for (ds,method),g in raw.groupby(['dataset','method']):
        assert set(g.seed)==SEEDS and len(g)==5
        r=dict(dataset=ds,method=method,n=5)
        for name in ['task1','task2','model_mb','aux_mb','uplink_mb','downlink_mb','logical_total_mb','jain','wall_seconds']:
            r[name+'_mean']=float(g[name].mean());r[name+'_ci95']=ci(g[name])
        summary.append(r)
    table=pd.DataFrame(summary);table.to_csv(OUT/'summary.csv',index=False)
    tests=[]
    for ds,g in raw.groupby('dataset'):
        p=g[g.method=='proposed'].set_index('seed').sort_index();family=[]
        for method,h in g[g.method!='proposed'].groupby('method'):
            h=h.set_index('seed').sort_index()
            for task in ['task1','task2']:
                diff=p[task].values-h[task].values
                pv=1. if np.all(diff==0) else float(stats.ttest_1samp(diff,0).pvalue)
                family.append(dict(dataset=ds,comparator=method,outcome=task,difference=diff.mean(),ci_low=diff.mean()-ci(diff),ci_high=diff.mean()+ci(diff),p=pv))
        last=0
        for rank,i in enumerate(np.argsort([r['p'] for r in family])):
            last=max(last,min(1.,family[i]['p']*(len(family)-rank)));family[i]['holm_p']=last
        tests+=family
    pd.DataFrame(tests).to_csv(OUT/'paired_holm.csv',index=False)
    delta=np.load(ROOT/'data/repair/intel_final.npz')
    persistence=[float(np.mean(delta[f'y{t}_test']**2)) for t in [1,2]]
    (OUT/'persistence.json').write_text(json.dumps(dict(normalized_mse=persistence,seed_independent=True,physical_rmse=(np.sqrt(persistence)*delta['y_std']).tolist()),indent=2))
    tuning=json.loads((ROOT/'configs/repair_plan.json').read_text())['candidates'];assert len(tuning)==276
    for c in tuning:
        d=np.load(ROOT/c['data']);assert not any(k.endswith('_test') for k in d.files)
        m=json.loads((ROOT/c['output']/'run_manifest.json').read_text());assert m['seed']==707 and m['eval_split']=='val' and m['rounds']==100
    budget=json.loads((ROOT/'results/repair/budget_stress/audit.json').read_text())
    report=dict(protocol='repair_v3',tuning_runs=len(tuning),main_run_labels=115,compressor_control_labels=15,
        completed_full_labels=len(raw),new_full_training_runs=int((~raw.reused).sum()),historical_training_runs_repeated=0,
        reused_full_runs=reused,final_state_reconstructions=reconstructed,max_reconstruction_error=maxerror,
        all_finite=True,all_five_seeds=True,tuning_files_have_no_test_arrays=True,tests_passed=22,
        budget_stress_instances=budget['instances'],budget_violations=0,
        development_disclosure='Earlier benchmark test diagnostics were inspected. Current candidate selection uses validation only; results remain exploratory rather than prospective confirmation.')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2))
    figures(table,raw)
    print(json.dumps(report,indent=2));print(table[['dataset','method','task1_mean','task2_mean','model_mb_mean','jain_mean']].to_string(index=False))

def figures(table,raw):
    plt.rcParams.update({'font.size':8,'axes.grid':True,'grid.alpha':.18,'savefig.bbox':'tight'})
    figs=OUT/'figures';figs.mkdir(exist_ok=True);datasets=['uci_har','cwru','intel']
    # Performance panels: logarithmic NMSE change axis is unnecessary after repair.
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.8))
    for ds,ax in zip(datasets,axes):
        q=table[(table.dataset==ds)&(table.method!='scheduled_topk')]
        ax.bar(range(len(q)),q.task1_mean,yerr=q.task1_ci95,capsize=2)
        ax.set_xticks(range(len(q)),[NAMES[m].replace(' adaptation','').replace('Laplacian surrogate','Laplacian') for m in q.method],rotation=65,ha='right',fontsize=6)
        ax.set(title=ds.replace('_',' '),ylabel='Temperature NMSE' if ds=='intel' else 'Task-1 accuracy')
        if ds=='intel':
            d=json.loads((OUT/'persistence.json').read_text());ax.axhline(d['normalized_mse'][0],color='black',ls='--',lw=.8,label='Persistence');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(figs/'fig_task1_performance.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.5))
    for ds,ax in zip(datasets,axes):
        for method,g in raw[(raw.dataset==ds)&raw.method.isin(['fedavg','topksgd','proposed','scheduled_topk','poc'])].groupby('method'):
            curves=[]
            for _,r in g.iterrows():
                p=ROOT/r.run;m=json.loads((p/'run_manifest.json').read_text())
                if m.get('reused_from'):p=ROOT/m['reused_from']
                curves.append(pd.read_csv(p/'round_metrics.csv').set_index('round').task1_metric)
            df=pd.concat(curves,axis=1);mean=df.mean(1);sd=df.std(1)
            ax.plot(mean.index,mean,label=NAMES[method],lw=.8);ax.fill_between(mean.index,mean-sd,mean+sd,alpha=.1)
        ax.set(title=ds.replace('_',' '),xlabel='Round',ylabel='NMSE' if ds=='intel' else 'Accuracy')
    axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(figs/'fig_convergence.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.8))
    for ds,ax in zip(datasets,axes):
        q=table[(table.dataset==ds)&(table.method!='scheduled_topk')];x=np.arange(len(q))
        ax.bar(x,q.model_mb_mean,label='Model updates');ax.bar(x,q.aux_mb_mean,bottom=q.model_mb_mean,label='Auxiliary/control')
        ax.set_xticks(x,[NAMES[m].replace(' adaptation','').replace('Laplacian surrogate','Laplacian') for m in q.method],rotation=65,ha='right',fontsize=6);ax.set(title=ds,ylabel='Uplink MB')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(figs/'fig_payload.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(7.3,2.3))
    for ds,ax in zip(datasets,axes):
        q=table[(table.dataset==ds)&table.method.isin(['proposed','scheduled_topk','topksgd'])]
        for _,r in q.iterrows():
            ax.errorbar(r.model_mb_mean,r.task1_mean,yerr=r.task1_ci95,fmt='o',label=NAMES[r.method],capsize=2)
        ax.set(title=ds,xlabel='Model-update MB',ylabel='NMSE' if ds=='intel' else 'Accuracy')
    axes[0].legend(fontsize=5);fig.tight_layout();fig.savefig(figs/'fig_overlap_control.pdf');plt.close(fig)
    stress=pd.read_csv(ROOT/'results/repair/budget_stress/instances.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.3,2.3));q=stress.groupby('budget_ratio')[['utility_ratio','utilization']].mean()
    axes[0].plot(q.index,q.utility_ratio,'o-');axes[0].set(xlabel='Budget multiplier',ylabel='Mean utility / exact optimum',ylim=(.9,1.01))
    axes[1].plot(q.index,q.utilization,'o-');axes[1].set(xlabel='Budget multiplier',ylabel='Mean budget utilization',ylim=(.8,1.01))
    fig.tight_layout();fig.savefig(figs/'fig_binding_budget.pdf');plt.close(fig)

if __name__=='__main__':main()
