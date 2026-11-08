"""Repair PoC candidate-model delivery counters without any model training.

Raw pre-correction records remain in provenance/poc_accounting_before.tar.gz.
Current manifests retain the actual training-source hash and config identity.
"""
from pathlib import Path
import sys,json,hashlib,tarfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.communication_accounting import model_recipients

def main():
    configs=json.loads((ROOT/'configs/repair_plan.json').read_text())['candidates']
    for name in ['repair_final_plan.json','repair_control_plan.json','repair_architecture_plan.json']:
        configs+=json.loads((ROOT/'configs'/name).read_text())
    affected=[c for c in configs if c['method']=='poc']
    original=ROOT/'provenance/poc_accounting_before.tar.gz'
    if not original.exists():
        with tarfile.open(original,'w:gz') as archive:
            for c in affected:
                for name in ['run_manifest.json','round_metrics.csv']:
                    p=ROOT/c['output']/name
                    archive.add(p,arcname=str(p.relative_to(ROOT)))
    ledger=[]
    for c in affected:
        folder=ROOT/c['output'];p=folder/'run_manifest.json'
        m=json.loads(p.read_text());selected=min(24,m['clients'])
        recipients=model_recipients('poc',selected,m['clients'])
        per_round=recipients*m['parameters']*4
        previous=m.get('accounting_correction',{}).get('original_downlink_bytes',m['logical_downlink_bytes'])
        expected=per_round*m['rounds']
        assert previous==selected*m['parameters']*4*m['rounds']
        correction=dict(version='candidate_downlink_v1',original_downlink_bytes=previous,
            corrected_downlink_bytes=expected,model_recipients_per_round=recipients,
            reason='All PoC candidates receive the current global model before local loss reporting.',
            training_repeated=False,training_source_hash_preserved=True)
        m['logical_downlink_bytes']=expected;m['accounting_correction']=correction
        p.write_text(json.dumps(m,indent=2))
        q=folder/'round_metrics.csv';curve=pd.read_csv(q)
        curve['logical_downlink_bytes']=curve['round']*per_round
        curve.to_csv(q,index=False)
        ledger.append(dict(run=c['output'],**correction))
    out=ROOT/'results/repair/analysis';out.mkdir(exist_ok=True,parents=True)
    result=dict(corrected_runs=len(ledger),training_runs_repeated=0,
        immutable_original_archive=str(original.relative_to(ROOT)),
        original_archive_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),records=ledger)
    (out/'communication_correction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
