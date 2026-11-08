"""Twelve validation candidates per method/dataset, then frozen final fits.

The final refit phase never selects settings from test outcomes. Earlier
diagnostic and archived outputs remain immutable; exact historical settings are
reused where their semantics match rather than repeated.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json,copy
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.repair import train as original_train
from src.scaffold_weighted import train as weighted_scaffold_train

def train(config):
    return weighted_scaffold_train(config) if config["method"]=="scaffold" else original_train(config)

SEEDS=[11,22,33,44,55]
METHODS=['fedavg','fedprox','fmtl','scaffold','topksgd','poc','proposed']
def candidates(method):
    if method in ['fedavg','scaffold','topksgd','poc']:
        return [dict(lr=lr,steps=steps) for lr in [.0003,.001,.003,.01,.03,.1] for steps in [2,4]]
    if method in ['fedprox','fmtl']:
        key='mu' if method=='fedprox' else 'coupling'
        return [dict(lr=lr,steps=2,**{key:value}) for lr in [.001,.01,.1] for value in [.001,.01,.1,1.]]
    if method=='proposed':
        return [dict(lr=lr,steps=2,rho=.3,sigma=sigma,fairness_mix=mix) for lr in [.001,.01,.1] for sigma in [0.,.6] for mix in [0.,.25]]
    if method=='fedrep':
        return [dict(lr=lr,head_steps=h,rep_steps=r) for lr in [.001,.01,.1] for h in [1,3] for r in [1,2]]
    if method=='ifca':
        return [dict(lr=lr,steps=s,clusters=k) for lr in [.001,.01,.1] for s in [2,4] for k in [2,3]]

def reuse(conf):
    ds=conf['dataset_name'];m=conf['method'];seed=conf['seed']
    # Only unchanged classification settings have a historical match. SCAFFOLD
    # has a new round-snapshot correction, and Intel has new residual targets.
    if ds=='intel' or m=='scaffold' or conf.get('lr')!=.01 or conf.get('steps',2)!=2:return None
    path=None
    if m in ['fedavg','topksgd']:path=ROOT/f'results/full/{ds}_{m}_{seed}'
    if m=='fedprox' and conf.get('mu')==.01:path=ROOT/f'results/full/{ds}_{m}_{seed}'
    if m=='fmtl' and conf.get('coupling')==.01:path=ROOT/f'results/full/{ds}_{m}_{seed}'
    if m=='poc':path=ROOT/f'results/supplementary/{ds}_poc_mix0_{seed}'
    if m=='proposed' and conf.get('sigma')==.6:
        mix=conf.get('fairness_mix',0.)
        path=ROOT/f'results/v2/main/{ds}_proposed_{seed}' if mix==0 else ROOT/f'results/supplementary/{ds}_proposed_mix{mix}_{seed}'
    if path is not None and (path/'run_manifest.json').exists():
        import pandas as pd
        hist=json.loads((path/'run_manifest.json').read_text());curve=pd.read_csv(path/'round_metrics.csv');final=curve.iloc[-1]
        assert hist['rounds']==300
        # New auxiliary/downlink accounting can be reconstructed algebraically
        # without repeating training; scalar norm reports were previously omitted.
        rec=dict(**conf,reused_from=str(path.relative_to(ROOT)),final_metrics=[float(final.task1_metric),float(final.task2_metric)],
                 payload_bytes=hist['payload_bytes'],selection_report_bytes=57600 if m in ['poc','proposed'] else 0,
                 logical_downlink_bytes=(48 if m=='poc' else 24)*300*hist['parameters']*4,parameters=hist['parameters'],clients=80,
                 wall_seconds=hist['wall_seconds'],source_sha256='historical-protocol-reuse',protocol='repair_v3 historical semantically identical classification setting')
        part=pd.read_csv(path/'participation.csv')['rounds'].to_numpy();rec['jain']=float(part.sum()**2/(len(part)*(part**2).sum()))
        target=ROOT/conf['output'];target.mkdir(parents=True,exist_ok=True)
        (target/'run_manifest.json').write_text(json.dumps(rec,indent=2));print('HISTORICAL REUSE',target,flush=True);return rec
    return None

def execute(conf):
    if conf['eval_split']=='test':
        cached=reuse(conf)
        if cached is not None:return cached
    return train(conf)

def main():
    os.chdir(ROOT)
    tune=[]
    for ds in ['uci_har','cwru','intel']:
        for method in METHODS+(['fedrep','ifca'] if ds=='intel' else []):
            grid=candidates(method);assert len(grid)==12
            for i,hp in enumerate(grid):
                tune.append(dict(dataset_name=ds,data=f'data/repair/{ds}_tune.npz',method=method,seed=707,rounds=100,
                    eval_split='val',zero_heads=ds=='intel',**hp,
                    **({'control_weighting':'population_sample_weights'} if method=='scaffold' else {}),
                    output=f'results/repair/scaffold_weighted/tuning/{ds}_{method}_{i:02}' if method=='scaffold' else f'results/repair/tuning/{ds}_{method}_{i:02}'))
    (ROOT/'configs/repair_plan.json').write_text(json.dumps(dict(protocol='repair_v3',trials=12,rounds=100,selection='Within joint error tolerance .0025 classification / 0 regression of best validation score; then minimize update bytes; then maximize Jain; then minimize validation error. No test-guided selection.',candidates=tune),indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        tuned=list(pool.map(train,tune));final=[];selected={}
        for ds in ['uci_har','cwru','intel']:
            selected[ds]={}
            for method in METHODS+(['fedrep','ifca'] if ds=='intel' else []):
                records=[r for r in tuned if r['dataset_name']==ds and r['method']==method]
                def error(r):return sum(r['final_metrics'])/2 if ds=='intel' else 1-sum(r['final_metrics'])/2
                best=min(error(r) for r in records)
                feasible=[r for r in records if error(r)<=best+(0 if ds=='intel' else .0025)]
                winner=min(feasible,key=lambda r:(r['payload_bytes'],-r['jain'],error(r)))
                hp={k:winner[k] for k in ['lr','steps','mu','coupling','sigma','rho','fairness_mix','clusters','head_steps','rep_steps'] if k in winner}
                selected[ds][method]=dict(hyperparameters=hp,validation_metrics=winner['final_metrics'],validation_run=next(c['output'] for c in tune if c['dataset_name']==ds and c['method']==method and all(c.get(k)==v for k,v in hp.items())))
                for seed in SEEDS:
                    final.append(dict(dataset_name=ds,data=f'data/repair/{ds}_final.npz',method=method,seed=seed,rounds=300,
                        eval_split='test',zero_heads=ds=='intel',**hp,
                        **({'control_weighting':'population_sample_weights'} if method=='scaffold' else {}),
                        output=f'results/repair/scaffold_weighted/final/{ds}_{method}_{seed}' if method=='scaffold' else f'results/repair/final/{ds}_{method}_{seed}'))
        (ROOT/'results/repair/selected.json').write_text(json.dumps(selected,indent=2))
        (ROOT/'configs/repair_final_plan.json').write_text(json.dumps(final,indent=2))
        list(pool.map(execute,final))
        # Isolate overlap using exactly the selected scheduling configuration.
        matched=[]
        for ds in ['uci_har','cwru','intel']:
            hp=copy.deepcopy(selected[ds]['proposed']['hyperparameters']);hp['sigma']=1.
            for seed in SEEDS:
                matched.append(dict(dataset_name=ds,data=f'data/repair/{ds}_final.npz',method='proposed',seed=seed,rounds=300,
                    eval_split='test',zero_heads=ds=='intel',role='scheduled_topk_control',**hp,
                    output=f'results/repair/controls/{ds}_scheduled_topk_{seed}'))
        (ROOT/'configs/repair_control_plan.json').write_text(json.dumps(matched,indent=2));list(pool.map(train,matched))

if __name__=='__main__':main()
