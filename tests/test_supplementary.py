import tempfile, unittest
from pathlib import Path
import numpy as np
from src.supplementary import train, evaluate
from src.experiment import Layout, loss_gradient
from src.preprocess_intel_v2 import preprocess
import gzip

class SupplementTests(unittest.TestCase):
    def test_new_methods_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(4)
            x=rng.normal(size=(200,4)).astype('float32');y=x[:,0].astype('float32')
            data={}
            for split in ('train','val','test'):
                data.update({f'x_{split}':x,f'y1_{split}':y,f'y2_{split}':-y,f'client_{split}':np.repeat(np.arange(10),20)})
            data['task_type']=np.array(['regression','regression'])
            p=root/'toy.npz';np.savez(p,**data)
            for method in ('fedavg','topksgd','proposed','poc','fedrep','ifca'):
                conf=dict(data=str(p),method=method,seed=11,rounds=2,output=str(root/method))
                m=train(conf);self.assertTrue(np.isfinite(m['final_metrics']).all());self.assertEqual(train(conf)['config_sha256'],m['config_sha256'])
                state=np.load(root/method/'final_state.npz')
                if method=='fedrep':
                    self.assertGreater(np.std(state['personal_heads'],axis=0).max(),0)

    def test_true_timestamps_and_purged_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'raw.gz';out=Path(tmp)/'data.npz'
            with gzip.open(source,'wt') as f:
                for i in range(120):
                    # epoch advances by one; wall clock advances by 30 seconds.
                    for client in (1,2):
                        f.write(f'2004-03-01 {i//120:02}:{i//2:02}:{(i%2)*30:02} {i} {client} {20+i/100} 40 100 2.5\n')
            preprocess(source,out);d=np.load(out)
            self.assertGreaterEqual(d['x_mean'][4],30)
            self.assertLess(d['x_mean'][4],31)
            self.assertLessEqual(d['target_timestamp_train'].max(),d['timestamp_val'].min())
            self.assertLessEqual(d['target_timestamp_val'].max(),d['timestamp_test'].min())

    def test_regression_gradient_finite_difference(self):
        rng=np.random.default_rng(9);layout=Layout(3,2,(1,1),('regression','regression'))
        v=rng.normal(scale=.1,size=layout.parameters);x=rng.normal(size=(8,3));y=rng.normal(size=8)
        _,grad=loss_gradient(v,layout,x,y,0)
        for j in layout.task_indices[0]:
            plus=v.copy();minus=v.copy();plus[j]+=1e-5;minus[j]-=1e-5
            numerical=(loss_gradient(plus,layout,x,y,0)[0]-loss_gradient(minus,layout,x,y,0)[0])/2e-5
            self.assertAlmostEqual(grad[j],numerical,places=5)
