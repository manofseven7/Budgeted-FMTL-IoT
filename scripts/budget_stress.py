"""Controlled binding-budget stress test; no physical-energy interpretation."""
from pathlib import Path
import itertools,sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.core import Candidate,select_budgeted_clients

def main():
    rng=np.random.default_rng(1703);n=16;cap=8;floor=4
    combinations=[c for k in range(floor,cap+1) for c in itertools.combinations(range(n),k)]
    masks=np.zeros((len(combinations),n),dtype='float64')
    for j,c in enumerate(combinations):masks[j,list(c)]=1
    records=[]
    for trial in range(200):
        costs=rng.uniform(.5,2,n);scores=rng.uniform(.05,1,n)
        exact_cost=masks@costs;exact_utility=masks@scores
        for ratio in [.65,.8,1.]:
            budget=max(float(np.sort(costs)[:floor].sum()+1e-9),float(np.sort(costs)[:cap].sum()*ratio))
            feasible=exact_cost<=budget+1e-12;optimum=float(exact_utility[feasible].max())
            selected,spent=select_budgeted_clients([Candidate(i,float(scores[i]),float(costs[i])) for i in range(n)],budget,floor,cap)
            assert floor<=len(selected)<=cap and spent<=budget+1e-12
            records.append(dict(trial=trial,budget_ratio=ratio,budget=budget,spent=spent,utilization=spent/budget,
                selected=len(selected),utility=float(scores[selected].sum()),optimal_utility=optimum,
                utility_ratio=float(scores[selected].sum()/optimum),floor_feasible=True,
                cap_infeasible=bool(np.sort(costs)[:cap].sum()>budget)))
    out=ROOT/'results/repair/budget_stress';out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(records);frame.to_csv(out/'instances.csv',index=False)
    frame.groupby('budget_ratio')[['utilization','selected','utility_ratio','cap_infeasible']].mean().to_csv(out/'summary.csv')
    (out/'audit.json').write_text(json.dumps(dict(instances=len(frame),budget_violations=0,floor_violations=0,
        exact_enumerated_sets=len(masks),seed=1703,cost_scope='Controlled heterogeneous synthetic costs, not measured energy',
        interpretation='Finite-instance heuristic check; no approximation guarantee'),indent=2))
    print(frame.groupby('budget_ratio')[['utilization','selected','utility_ratio','cap_infeasible']].mean())

if __name__=='__main__':main()
