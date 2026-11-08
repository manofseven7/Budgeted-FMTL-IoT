"""Reconstruct all new architecture-control metrics, preserving old outputs."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.repair import evaluate
from src.experiment import Layout
def main():
    plan=json.loads((ROOT/'configs/repair_architecture_plan.json').read_text());d=np.load(ROOT/'data/repair/intel_final.npz')
    layout=Layout(6,16,(1,1),('regression','regression'));records=[];error=0
    for conf in plan:
        p=ROOT/conf['output'];m=json.loads((p/'run_manifest.json').read_text());s=np.load(p/'final_state.npz')
        personal=np.zeros((len(s['clients']),layout.parameters),dtype='float32')
        if m['method']=='fedrep':personal[:,layout.shared.size:]=s['personal_heads']
        actual=evaluate(s['models'],layout,d,'test',m['method'],personal,s['cluster'],s['clients'])
        error=max(error,float(np.max(np.abs(np.array(actual)-m['final_metrics']))));assert error<1e-7
        old=json.loads((ROOT/f"results/supplementary/intel_v2_{m['method']}_{m['seed']}/run_manifest.json").read_text())
        assert old['lr']==m['lr']==.03 and old['rounds']==m['rounds']==300
        records.append(dict(method=m['method'],seed=m['seed'],old_temperature=old['final_metrics'][0],new_temperature=actual[0],old_humidity=old['final_metrics'][1],new_humidity=actual[1]))
    out=ROOT/'results/repair/analysis';frame=pd.DataFrame(records);frame.to_csv(out/'architecture_control_seed_level.csv',index=False)
    frame.groupby('method').mean(numeric_only=True).drop(columns='seed').to_csv(out/'architecture_control_summary.csv')
    report=dict(new_parameterization_fits=len(records),old_absolute_fits_reused=len(records),old_training_repeated=0,max_reconstruction_error=error,
        interpretation='Common persistence-offset and zero-head parameterization control at fixed optimizer settings; not a proposed-only benefit.')
    (out/'architecture_control_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
