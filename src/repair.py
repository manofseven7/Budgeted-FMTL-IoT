"""Reviewer supplements. Original archived runs are never overwritten.

FedRep and IFCA are explicit two-task adaptations, not official benchmark
reproductions. They are evaluated only on Intel's persistent sensor clients.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time
import numpy as np
from .experiment import Layout, initialise, loss_gradient, local_update, metric, partition, _write_csv
from .core import Candidate, select_budgeted_clients, compress_with_error_feedback, topk_support, sparse_payload_bytes

def evaluate(models, layout, data, split, method, personal, cluster, parts):
    x=data[f'x_{split}']; ys=[data[f'y{i}_{split}'] for i in (1,2)]
    if method not in ('fedrep','ifca'):
        return [metric(models[0],layout,x,ys[t],t) for t in (0,1)]
    ids=data[f'client_{split}']; result=np.zeros(2); total=0
    for client,identifier in enumerate(parts):
        ind=np.flatnonzero(ids==identifier)
        if not len(ind): continue
        model=models[cluster[client]].copy() if method=='ifca' else models[0].copy()
        if method=='fedrep': model[layout.shared.size:]=personal[client,layout.shared.size:]
        result += len(ind)*np.array([metric(model,layout,x[ind],ys[t][ind],t) for t in (0,1)])
        total+=len(ind)
    if total!=len(x): raise ValueError('Unseen test clients cannot use personalized heads without an adaptation protocol')
    return (result/total).tolist()

from .communication_accounting import model_recipients, accounting_compatible_resume

def train(config):
    out=Path(config['output']); data_path=Path(config['data'])
    identity={k:v for k,v in config.items() if k!='output'}
    identity['data_sha256']=hashlib.sha256(data_path.read_bytes()).hexdigest()
    identity['source_sha256']=hashlib.sha256(b''.join(Path(__file__).with_name(n).read_bytes() for n in ['repair.py','experiment.py','core.py'])).hexdigest()
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    manifest_path=out/'run_manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old.get('config_sha256')!=key and not accounting_compatible_resume(old, identity): raise ValueError(f'Conflicting existing run {out}')
        print('REUSE',str(out),flush=True); return old
    data=np.load(data_path); method=config['method']; seed=config['seed']; lr=config.get('lr',.01)
    rng=np.random.default_rng(seed)
    types=tuple(data['task_type']) if 'task_type' in data else ('classification','classification')
    x=data['x_train']; ys=[data['y1_train'],data['y2_train']]
    layout=Layout(x.shape[1],16,tuple(1 if types[t]=='regression' else int(ys[t].max()+1) for t in (0,1)),types)
    base=initialise(layout,rng)
    if config.get('zero_heads',False):base[layout.shared.size:]=0
    if 'client_train' in data:
        identifiers=np.unique(data['client_train']); parts=[np.flatnonzero(data['client_train']==i) for i in identifiers]
    else:
        parts=partition(ys[0],80,.3,rng); identifiers=np.arange(80)
    n=len(parts); cap=min(24,n)
    models=np.stack([base.copy() for _ in range(config.get('clusters',3) if method=='ifca' else 1)])
    if method=='ifca':
        for i in range(1,len(models)):
            models[i]=initialise(layout,rng)
            if config.get('zero_heads',False):models[i,layout.shared.size:]=0
    personal=np.tile(base,(n,1)); cluster=np.zeros(n,dtype=int)
    residual=np.zeros((n,2,layout.parameters),dtype='float32')
    participation=np.zeros(n,dtype=int); reserve=rng.uniform(.8,1.2,n); norms=np.ones((n,2))
    controls=np.zeros((n,2,layout.parameters),dtype='float32');server_control=np.zeros((2,layout.parameters),dtype='float32')
    rows=[]; total_payload=0; selection_bytes=0;downlink_bytes=0; start=time.perf_counter()
    for r in range(1,config['rounds']+1):
        if method=='proposed':
            score=(np.clip(reserve/1.2,0,1)+norms.mean(1)/(norms.mean(1).max()+1e-12))/3
            mix=config.get('fairness_mix',0.)
            # A controlled fairness/utility score interpolation. All costs .025.
            fair=1-(participation-participation.min())/(participation.max()-participation.min()+1)
            score=(1-mix)*score+mix*fair
            chosen,_=select_budgeted_clients([Candidate(i,float(score[i]),.025) for i in range(n)],.025*cap*1.05,8,cap)
        elif method=='poc':
            candidates=rng.choice(n,min(2*cap,n),replace=False)
            losses=[]
            for c in candidates:
                ind=parts[c][:64]
                losses.append(sum(loss_gradient(models[0],layout,x[ind],ys[t][ind],t)[0] for t in (0,1))/2)
            chosen=[int(candidates[i]) for i in np.argsort(-np.array(losses),kind='stable')[:cap]]
            selection_bytes+=4*len(candidates) # float32 candidate loss reports; model delivery counted below
        else: chosen=rng.choice(n,cap,replace=False).tolist()
        updates=[[] for _ in models]; weights=[[] for _ in models]; payload=0
        snapshot=server_control.copy();deltas=[];similarities=[]
        if method=='fedrep':downlink_bytes+=cap*layout.shared.size*4
        else:downlink_bytes+=model_recipients(method,cap,n)*layout.parameters*4*len(models)
        if method=='scaffold':downlink_bytes+=cap*sum(len(v) for v in layout.task_indices)*4
        if method=='proposed':selection_bytes+=cap*8
        for c in chosen:
            participation[c]+=1; ind=parts[c]
            if method=='ifca':
                probe=ind[:64]
                losses=[sum(loss_gradient(m,layout,x[probe],ys[t][probe],t)[0] for t in (0,1)) for m in models]
                cluster[c]=int(np.argmin(losses))
            k=cluster[c] if method=='ifca' else 0
            model=models[k]
            if method=='fedrep':
                local=model.copy(); local[layout.shared.size:]=personal[c,layout.shared.size:]
                # Alternating one head step then one representation step per task.
                for phase in ['head','representation']:
                    for t in (0,1):
                        for _ in range(config.get('head_steps',1) if phase=='head' else config.get('rep_steps',1)):
                            batch=rng.choice(ind,min(32,len(ind)),replace=False)
                            _,g=loss_gradient(local,layout,x[batch],ys[t][batch],t)
                            active=np.setdiff1d(layout.task_indices[t],layout.shared) if phase=='head' else layout.shared
                            local[active]-=lr*g[active]
                personal[c]=local
                update=np.zeros_like(model); update[layout.shared]=model[layout.shared]-local[layout.shared]
                payload+=layout.shared.size*4
            else:
                raw=[];client_delta=[]
                for t in (0,1):
                    u,g=local_update(model,layout,x,ys[t],ind,t,rng,config.get('steps',2),32,lr,
                        proximal=config.get('mu',.01) if method=='fedprox' else 0.,
                        control=snapshot[t]-controls[c,t] if method=='scaffold' else None,
                        fmtl_strength=config.get('coupling',.01) if method=='fmtl' else 0.)
                    if method=='scaffold':
                        active=layout.task_indices[t];updated=controls[c,t].copy()
                        updated[active]=controls[c,t,active]-snapshot[t,active]+u[active]/(config.get('steps',2)*lr)
                        client_delta.append(updated-controls[c,t]);controls[c,t]=updated
                    raw.append(u); norms[c,t]=np.linalg.norm(g)
                if method=='proposed':
                    sent,residual[c],masks,sim=compress_with_error_feedback(raw,residual[c],config.get('rho',.3),config.get('sigma',.6))
                    similarities.append(float(sim[0,1]))
                    payload+=sum(sparse_payload_bytes(m) for m in masks)
                elif method=='topksgd':
                    sent=[]
                    for t in (0,1):
                        comp=raw[t]+residual[c,t]; mask=topk_support(comp,config.get('rho',.3))
                        u=np.where(mask,comp,0).astype('float32');residual[c,t]=comp-u
                        sent.append(u);payload+=sparse_payload_bytes(mask)
                else:
                    sent=raw;payload+=sum(len(layout.task_indices[t])*4 for t in (0,1))
                    if method=='ifca': selection_bytes+=4 # cluster identifier
                    if method=='scaffold':selection_bytes+=sum(len(v) for v in layout.task_indices)*4
                update=.5*(sent[0]+sent[1])
            updates[k].append(update);weights[k].append(len(ind))
            if method=='scaffold':deltas.append(client_delta)
            reserve[c]=max(0,reserve[c]-.025)
        for k in range(len(models)):
            if updates[k]: models[k]-=np.average(updates[k],axis=0,weights=weights[k])
        if method=='scaffold':server_control+=np.sum(deltas,axis=0)/n
        total_payload+=payload
        # No test evaluation during tuning; full runs evaluate fixed checkpoints.
        if r%10==0 or r==config['rounds']:
            if method=='ifca':
                for c,ind in enumerate(parts):
                    probe=ind[:64]
                    cluster[c]=np.argmin([sum(loss_gradient(m,layout,x[probe],ys[t][probe],t)[0] for t in (0,1)) for m in models])
            values=evaluate(models,layout,data,config.get('eval_split','test'),method,personal,cluster,identifiers)
            rows.append(dict(round=r,task1_metric=values[0],task2_metric=values[1],cumulative_payload_bytes=total_payload,
                             selection_report_bytes=selection_bytes,logical_downlink_bytes=downlink_bytes,active_clients=cap,
                             mean_task_similarity=float(np.mean(similarities)) if similarities else 0.))
    out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'round_metrics.csv',rows)
    _write_csv(out/'participation.csv',[dict(client=int(identifiers[c]),rounds=int(v)) for c,v in enumerate(participation)])
    np.savez_compressed(out/'final_state.npz',models=models,personal_heads=personal[:,layout.shared.size:] if method=='fedrep' else np.empty(0),
                        cluster=cluster,clients=identifiers,server_control=server_control)
    manifest=dict(**identity,config_sha256=key,clients=n,clients_per_round=cap,parameters=layout.parameters,
                  payload_bytes=total_payload,selection_report_bytes=selection_bytes,logical_downlink_bytes=downlink_bytes,wall_seconds=time.perf_counter()-start,
                  jain=float(participation.sum()**2/(n*np.sum(participation**2))),final_metrics=values,
                  run_kind='validation_tuning' if config.get('eval_split')=='val' else 'reviewer_supplement',
                  freezing='Optional sensitivity mechanism excluded from active-method claims',
                  protocol='repair_v3; validation-selected hyperparameters; full auxiliary/control traffic accounted',
                  target='normalized residual relative to current observation' if config.get('zero_heads') else 'classification')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('DONE',out,values,flush=True);return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config',type=Path);args=ap.parse_args()
    train(json.loads(args.config.read_text()))
