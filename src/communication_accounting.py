"""Logical unicast accounting; optimizer updates are deliberately unchanged."""
from pathlib import Path
import hashlib
import json


def model_recipients(method, selected, clients):
    """PoC candidates need the current model before reporting local losses."""
    return min(2 * selected, clients) if method == 'poc' else selected


def accounting_compatible_resume(old, identity):
    """Accept only registered byte-counter changes, never altered fit settings."""
    path = Path(__file__).resolve().parents[1] / 'configs/accounting_source_compatibility.json'
    if not path.exists():
        return False
    allowed = json.loads(path.read_text())['source_pairs']
    if [old.get('source_sha256'), identity.get('source_sha256')] not in allowed:
        return False
    previous = dict(identity, source_sha256=old['source_sha256'])
    expected = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    return expected == old.get('config_sha256')
