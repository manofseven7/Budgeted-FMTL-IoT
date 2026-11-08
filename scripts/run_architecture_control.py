"""Isolate the sensor regression parameterization, reusing all absolute-model runs.

Only new residual/zero-head configurations are fitted. Old six-method five-seed
absolute forecasting outputs are read unchanged for the matched comparison.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.repair import train

def main():
    os.chdir(ROOT);configs=[]
    for method in ['fedavg','topksgd','proposed','poc','fedrep','ifca']:
        for seed in [11,22,33,44,55]:
            old=ROOT/f'results/supplementary/intel_v2_{method}_{seed}/run_manifest.json'
            m=json.loads(old.read_text());assert m['rounds']==300 and m['lr']==.03
            configs.append(dict(dataset_name='intel',data='data/repair/intel_final.npz',method=method,
                seed=seed,rounds=300,lr=.03,steps=2,clusters=3,head_steps=1,rep_steps=1,
                rho=.3,sigma=.6,fairness_mix=0.,zero_heads=True,eval_split='test',
                role='architecture_control',output=f'results/repair/architecture/intel_{method}_{seed}'))
    (ROOT/'configs/repair_architecture_plan.json').write_text(json.dumps(configs,indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(train,configs))

if __name__=='__main__':main()
