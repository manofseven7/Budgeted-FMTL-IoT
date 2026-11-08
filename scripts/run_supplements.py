"""Resume-only supplementary plan. Tuning consumes validation, never test.

New UCI/CWRU selection experiments retain the archived equalized .01 LR;
Intel timestamp-v2 methods each receive the same 3-point LR tuning budget.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import json, sys
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.supplementary import train

SEEDS=[11,22,33,44,55]
ROOT=Path(__file__).resolve().parents[1]

def main():
    os.chdir(ROOT)
    configs=[]
    for dataset in ['uci_har','cwru']:
        for method,mix in [('poc',0),('proposed',.25),('proposed',.5),('proposed',.75),('proposed',1.)]:
            for seed in SEEDS:
                configs.append(dict(data=f'data/processed/{dataset}.npz',method=method,seed=seed,rounds=300,lr=.01,
                    fairness_mix=mix,eval_split='test',output=f'results/supplementary/{dataset}_{method}_mix{mix}_{seed}'))
    methods=['fedavg','topksgd','proposed','poc','fedrep','ifca']
    tuning=[]
    for method in methods:
        for lr in [.003,.01,.03]:
            tuning.append(dict(data='data/processed/intel_timestamp_v2.npz',method=method,seed=707,rounds=60,lr=lr,
                               eval_split='val',output=f'results/tuning/intel_v2_{method}_lr{lr}'))
    Path('configs').mkdir(exist_ok=True)
    Path('configs/supplement_plan.json').write_text(json.dumps(dict(seeds=SEEDS,rounds=300,tuning=tuning,selection=configs),indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        tuned=list(pool.map(train,tuning))
        best={m:min([v for v in tuned if v['method']==m],key=lambda v:sum(v['final_metrics']))['lr'] for m in methods}
        Path('results/tuning/selected_learning_rates.json').write_text(json.dumps(best,indent=2))
        for method in methods:
            for seed in SEEDS:
                configs.append(dict(data='data/processed/intel_timestamp_v2.npz',method=method,seed=seed,rounds=300,
                    lr=best[method],eval_split='test',output=f'results/supplementary/intel_v2_{method}_{seed}'))
        for mix in [.25,.5,.75,1.]:
            for seed in SEEDS:
                configs.append(dict(data='data/processed/intel_timestamp_v2.npz',method='proposed',seed=seed,rounds=300,
                    lr=best['proposed'],fairness_mix=mix,eval_split='test',output=f'results/supplementary/intel_v2_proposed_mix{mix}_{seed}'))
        Path('configs/supplement_executed_plan.json').write_text(json.dumps(configs,indent=2))
        list(pool.map(train,configs))

if __name__=='__main__':main()
