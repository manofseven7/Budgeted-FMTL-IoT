"""Make SCAFFOLD's population control mean match sample-weighted objectives.

Preserves the superseded uniform-control calibration/fits and executes only new
weighted-control configurations; no completed training setting is repeated.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,sys,copy
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.scaffold_weighted import train

def main():
    os.chdir(ROOT);plan=json.loads((ROOT/'configs/repair_plan.json').read_text())
    if not (ROOT/'configs/repair_plan_uniform_control_history.json').exists():
        (ROOT/'configs/repair_plan_uniform_control_history.json').write_text(json.dumps(plan,indent=2))
        (ROOT/'configs/repair_final_plan_uniform_control_history.json').write_text((ROOT/'configs/repair_final_plan.json').read_text())
    originals=json.loads((ROOT/'configs/repair_plan_uniform_control_history.json').read_text())
    tuning=[]
    for c in originals['candidates']:
        if c['method']!='scaffold':continue
        c=copy.deepcopy(c);c['control_weighting']='population_sample_weights'
        c['output']=c['output'].replace('results/repair/tuning/','results/repair/scaffold_weighted/tuning/');tuning.append(c)
    with ProcessPoolExecutor(max_workers=4) as pool:
        records=list(pool.map(train,tuning));selected=json.loads((ROOT/'results/repair/selected.json').read_text());final=[]
        for ds in ['uci_har','cwru','intel']:
            values=[r for r in records if r['dataset_name']==ds]
            def error(r):return sum(r['final_metrics'])/2 if ds=='intel' else 1-sum(r['final_metrics'])/2
            best=min(error(r) for r in values);values=[r for r in values if error(r)<=best+(0 if ds=='intel' else .0025)]
            winner=min(values,key=lambda r:(r['payload_bytes'],-r['jain'],error(r)))
            hp={k:winner[k] for k in ['lr','steps']}
            selected[ds]['scaffold']=dict(hyperparameters=hp,validation_metrics=winner['final_metrics'],control_weighting='population_sample_weights')
            for seed in [11,22,33,44,55]:
                final.append(dict(dataset_name=ds,data=f'data/repair/{ds}_final.npz',method='scaffold',seed=seed,rounds=300,
                    eval_split='test',zero_heads=ds=='intel',control_weighting='population_sample_weights',**hp,
                    output=f'results/repair/scaffold_weighted/final/{ds}_scaffold_{seed}'))
        list(pool.map(train,final))
    current=json.loads((ROOT/'configs/repair_plan.json').read_text());current['candidates']=[c for c in current['candidates'] if c['method']!='scaffold']+tuning
    (ROOT/'configs/repair_plan.json').write_text(json.dumps(current,indent=2))
    finals=json.loads((ROOT/'configs/repair_final_plan.json').read_text());finals=[c for c in finals if c['method']!='scaffold']+final
    (ROOT/'configs/repair_final_plan.json').write_text(json.dumps(finals,indent=2));(ROOT/'results/repair/selected.json').write_text(json.dumps(selected,indent=2))
    print('Updated sample-weighted SCAFFOLD; 36 validation trials and 15 final fits; old convention preserved.',flush=True)

if __name__=='__main__':main()
