import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.repair import train
from src.core import cosine_similarity_matrix
from src.communication_accounting import model_recipients

ROOT=Path(__file__).resolve().parents[1]

class AccountingCorrectionTests(unittest.TestCase):
    def test_poc_candidate_delivery_preserves_training_and_cache(self):
        spec=importlib.util.spec_from_file_location('src._legacy_repair',ROOT/'provenance/pre_accounting_source/repair.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rng=np.random.default_rng(42)
            x=rng.normal(size=(160,3)).astype('float32');y=(x[:,0]>0).astype(int)
            data={}
            for split in ['train','test']:
                data.update({f'x_{split}':x,f'y1_{split}':y,f'y2_{split}':1-y,f'client_{split}':np.repeat(np.arange(80),2)})
            np.savez(p/'data.npz',**data)
            conf=dict(data=str(p/'data.npz'),method='poc',seed=11,rounds=2,lr=.01,output=str(p/'old'))
            legacy=old.train(conf)
            current=train(dict(conf,output=str(p/'new')))
            self.assertEqual(current['logical_downlink_bytes'],2*legacy['logical_downlink_bytes'])
            self.assertEqual(current['final_metrics'],legacy['final_metrics'])
            self.assertEqual(current['payload_bytes'],legacy['payload_bytes'])
            a=np.load(p/'old/final_state.npz');b=np.load(p/'new/final_state.npz')
            for k in a.files:np.testing.assert_array_equal(a[k],b[k])
            # Real historical keys hash the old source bytes. Restore that exact
            # identity to exercise registered accounting-only cache reuse.
            old_sha=json.loads((ROOT/'configs/accounting_source_compatibility.json').read_text())['source_pairs'][0][0]
            identity={k:v for k,v in conf.items() if k!='output'}
            identity.update(data_sha256=hashlib.sha256((p/'data.npz').read_bytes()).hexdigest(),source_sha256=old_sha)
            legacy.update(source_sha256=old_sha,config_sha256=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest())
            (p/'old/run_manifest.json').write_text(json.dumps(legacy))
            self.assertEqual(train(conf)['config_sha256'],legacy['config_sha256'])
            with self.assertRaises(ValueError):train(dict(conf,lr=.02))

    def test_candidate_count_and_full_vector_definition(self):
        self.assertEqual(model_recipients('poc',24,80),48)
        self.assertEqual(model_recipients('poc',24,30),30)
        self.assertEqual(model_recipients('proposed',24,80),24)
        vectors=[np.array([1.,10.,0.]),np.array([1.,0.,10.])]
        self.assertAlmostEqual(cosine_similarity_matrix(vectors)[0,1],1/101)
        self.assertLess(cosine_similarity_matrix(vectors)[0,1],.6)
        self.assertGreater(cosine_similarity_matrix([v[:1] for v in vectors])[0,1],.6)
