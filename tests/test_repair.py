import unittest,tempfile,json
from pathlib import Path
import numpy as np
from src.repair import train

class RepairTests(unittest.TestCase):
    def test_all_repaired_methods_finite_and_control_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rng=np.random.default_rng(3)
            arrays={}
            for split in ['train','val','test']:
                x=rng.normal(size=(200,4)).astype('float32')
                arrays.update({f'x_{split}':x,f'y1_{split}':x[:,0]*.001,f'y2_{split}':x[:,1]*.001,f'client_{split}':np.repeat(np.arange(10),20)})
            arrays['task_type']=np.array(['regression','regression']);np.savez(p/'d.npz',**arrays)
            results={}
            for m in ['fedavg','fedprox','fmtl','scaffold','topksgd','poc','proposed','fedrep','ifca']:
                conf=dict(data=str(p/'d.npz'),method=m,seed=11,rounds=2,lr=.01,zero_heads=True,output=str(p/m))
                r=train(conf);self.assertTrue(np.isfinite(r['final_metrics']).all());self.assertLess(max(r['final_metrics']),1e-5)
                self.assertEqual(r['config_sha256'],train(conf)['config_sha256']);results[m]=r
            # SCAFFOLD sends model updates and equally sized control updates.
            self.assertEqual(results['scaffold']['payload_bytes'],results['scaffold']['selection_report_bytes'])
            self.assertGreater(results['scaffold']['logical_downlink_bytes'],results['fedavg']['logical_downlink_bytes'])

    def test_residual_error_equals_absolute_forecast_error(self):
        rng=np.random.default_rng(5);y=rng.normal(size=100);current=rng.normal(size=100);correction=rng.normal(size=100)
        self.assertAlmostEqual(np.mean((correction-(y-current))**2),np.mean(((current+correction)-y)**2),places=13)

    def test_weighted_scaffold_control_population_identity(self):
        from src.scaffold_weighted import train as weighted_train
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rng=np.random.default_rng(9);x=rng.normal(size=(120,3)).astype('float32')
            # Unequal populations make an accidental uniform control mean fail.
            ids=np.repeat(np.arange(10),np.arange(1,11)*2);x=x[:len(ids)];y=x[:,0]*.01
            arrays={'task_type':np.array(['regression','regression'])}
            for split in ['train','test']:
                arrays.update({f'x_{split}':x,f'y1_{split}':y,f'y2_{split}':-y,f'client_{split}':ids})
            np.savez(p/'data.npz',**arrays)
            weighted_train(dict(data=str(p/'data.npz'),method='scaffold',seed=11,rounds=3,lr=.01,zero_heads=True,output=str(p/'run')))
            s=np.load(p/'run/final_state.npz')
            expected=np.einsum('n,ntp->tp',s['client_weights'],s['client_controls'])
            np.testing.assert_allclose(s['server_control'],expected,rtol=1e-5,atol=1e-9)
