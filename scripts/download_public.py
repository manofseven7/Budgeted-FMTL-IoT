"""Download provider bytes, validate containers, and record source SHA-256.

Existing validated downloads are reused. No dataset is redistributed in GitHub.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, json, urllib.request, zipfile, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess_public import CWRU_FILES
ROOT = Path(__file__).resolve().parents[1]

def fetch(item):
    url, relative = item
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        for attempt in range(5):
            try:
                with urllib.request.urlopen(url, timeout=90) as src, temporary.open('wb') as dst:
                    while chunk := src.read(1024 * 1024): dst.write(chunk)
                break
            except Exception:
                if attempt == 4: raise
                time.sleep(2 * (attempt + 1))
        temporary.replace(path)
    if path.suffix == '.mat':
        from scipy.io import loadmat
        assert any('DE_time' in k for k in loadmat(path)), path
    elif path.suffix == '.zip':
        with zipfile.ZipFile(path) as z: assert z.testzip() is None
    else:
        import gzip
        with gzip.open(path) as f:
            while f.read(1024 * 1024): pass
    result = dict(url=url, path=relative, bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    print(relative, result['bytes'], flush=True)
    return result

def main():
    ids = [i for groups in CWRU_FILES.values() for values in groups.values() for i in values]
    items = [('https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip','data/raw/uci_har.zip'),
             ('https://db.csail.mit.edu/labdata/data.txt.gz','data/raw/intel/data.txt.gz')]
    items += [(f'https://engineering.case.edu/sites/default/files/{i}.mat',f'data/raw/cwru/{i}.mat') for i in ids]
    with ThreadPoolExecutor(max_workers=6) as pool: records = list(pool.map(fetch, items))
    (ROOT/'data/raw/download_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    dest = ROOT/'data/extracted/uci_har'
    dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(ROOT/'data/raw/uci_har.zip') as z: z.extractall(dest)
    for path in dest.glob('*.zip'):
        with zipfile.ZipFile(path) as z: z.extractall(dest)

if __name__ == '__main__': main()
