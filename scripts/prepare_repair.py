"""Repair protocol fixed after the documented diagnostic failure.

Test outcomes from earlier development are disclosed as previously inspected.
All subsequent choices use the existing training/validation partitions only.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.preprocess_public import CWRU_FILES

def main():
    out=ROOT/'data/repair';out.mkdir(parents=True,exist_ok=True)
    reports=[]
    for ds in ['uci_har','cwru','intel']:
        source=ROOT/'data/processed'/('intel_timestamp_v2.npz' if ds=='intel' else ds+'.npz')
        d=dict(np.load(source));report=dict(dataset=ds,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        if ds=='intel':
            for split in ['train','val','test']:
                current=(d['x_'+split][:,:2]*d['x_std'][:2]+d['x_mean'][:2]-d['y_mean'])/d['y_std']
                for t in [1,2]:
                    d[f'absolute_y{t}_{split}']=d[f'y{t}_{split}'].copy()
                    d[f'baseline{t}_{split}']=current[:,t-1].astype('float32')
                    d[f'y{t}_{split}']=(d[f'y{t}_{split}']-current[:,t-1]).astype('float32')
            report.update(model='Persistence offset plus zero-head initialized neural residual; equal architecture for every FL comparator',split='existing purged chronological 64/16/20',test_status='Previously inspected in diagnostic v2; not represented as a new untouched holdout')
            np.savez_compressed(out/'intel_final.npz',**d)
            np.savez_compressed(out/'intel_tune.npz',**{k:v for k,v in d.items() if not k.endswith('_test')})
        else:
            x=d['x_train'].astype('float64')*d['feature_std']+d['feature_mean'];groups=d['group_train']
            if ds=='uci_har':val_ids=sorted(np.unique(groups))[-5:]
            else:val_ids=[ids[2] for g in CWRU_FILES.values() for ids in g.values()]
            val=np.isin(groups,val_ids);tr=~val
            mean=x[tr].mean(0);std=x[tr].std(0);std[std<1e-8]=1
            tune={}
            for name,mask in [('train',tr),('val',val)]:
                tune['x_'+name]=((x[mask]-mean)/std).astype('float32')
                for t in [1,2]:tune[f'y{t}_{name}']=d[f'y{t}_train'][mask]
                tune['group_'+name]=groups[mask]
            assert not set(tune['group_train'])&set(tune['group_val'])
            for t in [1,2]:assert set(tune[f'y{t}_train'])==set(tune[f'y{t}_val'])
            np.savez_compressed(out/f'{ds}_tune.npz',**tune)
            np.savez_compressed(out/f'{ds}_final.npz',**d)
            report.update(validation_groups=[int(v) for v in val_ids],inner_training=int(tr.sum()),validation=int(val.sum()),split='subject-disjoint inner split' if ds=='uci_har' else 'load-2 validation, loads-0/1 inner training; final refit loads-0/1/2',scaling='inner training only during tuning; official full training only during refit')
        reports.append(report)
    manifest=dict(protocol='repair_v3',development_disclosure='Post-hoc repair following earlier public-data diagnostics; no prospective confirmatory superiority claim.',
        tuning='12 candidate configurations per method and dataset; 100 rounds; seed707; joint mean validation error. No test arrays in tuning archives.',
        final='300 rounds; seeds11/22/33/44/55; matching completed semantic configurations reused where available',datasets=reports)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
