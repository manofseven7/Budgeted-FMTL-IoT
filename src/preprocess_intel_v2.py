"""Timestamp-correct Intel tasks, with purged chronological train/val/test.

Legacy preprocessing used epoch counters as seconds. Its results are retained
only as historical evidence. This version never overwrites that dataset.
"""
from pathlib import Path
import gzip, json, hashlib
import numpy as np
import pandas as pd

def preprocess(source, destination):
    source, destination = Path(source), Path(destination)
    with gzip.open(source, 'rt') as f:
        df = pd.read_csv(f, sep=r'\s+', names=['date','time','epoch','client','temperature','humidity','light','voltage'], on_bad_lines='skip')
    df['timestamp'] = pd.to_datetime(df['date']+' '+df['time'],errors='coerce')
    numeric = ['client','temperature','humidity','light','voltage']
    df[numeric] = df[numeric].apply(pd.to_numeric,errors='coerce')
    keep = np.isfinite(df[numeric]).all(axis=1) & df.timestamp.notna()
    keep &= df.client.between(1,54) & df.temperature.between(-10,60) & df.humidity.between(0,100) & (df.light>=0) & (df.voltage>=0)
    df = df[keep].sort_values(['client','timestamp']).drop_duplicates(['client','timestamp'])
    nxt = df.groupby('client').shift(-1)
    gap = (nxt.timestamp-df.timestamp).dt.total_seconds()
    keep = (gap>0)&(gap<=120)
    past_gap = df.groupby('client').timestamp.diff().dt.total_seconds().fillna(31).clip(0,120)
    cur, fut, gap = df[keep], nxt[keep], gap[keep]
    x = np.c_[cur[numeric[1:]].values,past_gap[keep].values,cur.client.values/54.]
    y = fut[['temperature','humidity']].values
    ts = cur.timestamp.astype('int64').values / 1e9
    future_ts = fut.timestamp.astype('int64').values / 1e9
    # Boundaries use only input timestamps; purge pairs crossing either one.
    cut1,cut2 = np.quantile(ts,[.64,.8])
    train = future_ts <= cut1
    val = (ts>cut1)&(future_ts<=cut2)
    test = ts>cut2
    xm,xs=x[train].mean(0),x[train].std(0); xs[xs<1e-8]=1
    ym,ys=y[train].mean(0),y[train].std(0); ys[ys<1e-8]=1
    x=((x-xm)/xs).astype('float32'); y=((y-ym)/ys).astype('float32')
    arrays=dict(x_mean=xm,x_std=xs,y_mean=ym,y_std=ys,task_type=np.array(['regression','regression']))
    for name,mask in [('train',train),('val',val),('test',test)]:
        arrays.update({f'x_{name}':x[mask],f'y1_{name}':y[mask,0],f'y2_{name}':y[mask,1],f'client_{name}':cur.client.values[mask].astype('int16'),f'timestamp_{name}':ts[mask],f'target_timestamp_{name}':future_ts[mask]})
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,**arrays)
    manifest=dict(protocol='intel_timestamp_v2',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_url='https://db.csail.mit.edu/labdata/labdata.html',boundaries_unix=[cut1,cut2],
                  counts={k:int(v.sum()) for k,v in [('train',train),('val',val),('test',test)]},
                  purged_pairs=int(len(ts)-train.sum()-val.sum()-test.sum()),maximum_gap_seconds=120,
                  clients={k:len(np.unique(arrays[f'client_{k}'])) for k in ['train','val','test']},
                  features=['temperature','humidity','light','voltage','previous_gap_seconds','sensor_id/54'],
                  gap_note='Only the previously observed gap is used as a predictor.',
                  scaling='training only; validation not refitted',dataset_sha256=hashlib.sha256(destination.read_bytes()).hexdigest())
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    preprocess('data/raw/intel/data.txt.gz','data/processed/intel_timestamp_v2.npz')
