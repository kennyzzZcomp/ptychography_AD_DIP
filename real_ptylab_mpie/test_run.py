"""Tiny synthetic tests only; never reconstruct the experimental H5."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mpie_entry', HERE/'run.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='_test_', dir=HERE)
        self.root = Path(self.tmp.name)
        self.source = self.root/'source.h5'
        yy, xx = np.indices((15,15))
        self.coords = np.stack([7.5 + yy.ravel()*.04, 7.7 + xx.ravel()*.04])
        y,x = np.indices((40,40))
        self.frames = np.stack([1+j*.01+y*.003+x*.005 for j in range(225)]).astype('float32')
        with h5py.File(self.source,'w') as f:
            f['ptychogramx'] = self.frames
            f['positionx'] = self.coords

    def tearDown(self):
        self.tmp.cleanup()

    def args(self,*extra):
        return entry.arguments(['--data',str(self.source),'--out',str(self.root/'out'),
                                '--detector-crop','32','--iters','2','--cpu',*extra])

    def test_conversion_and_frame_pairing(self):
        a = self.args()
        path,config = entry.prepare(a)
        crop = self.frames[:,4:36,4:36]
        with h5py.File(path,'r') as f:
            np.testing.assert_allclose(f['ptychogram'][()],crop/crop.mean(dtype='float64'),rtol=1e-6)
            expected = self.coords.T*1e-3
            expected -= expected.mean(axis=0)
            np.testing.assert_allclose(f['encoder'][()],expected,atol=1e-15)
            np.testing.assert_array_equal(f['original_frame_ids'][()],np.arange(225))
            self.assertAlmostEqual(float(f['dxd'][()]),3.76e-6)
            self.assertAlmostEqual(float(f['zo'][()]),.006)
        self.assertEqual(config['crop_slice_zero_based'],[4,36])
        self.assertFalse(config['TGV'])
        self.assertFalse(config['NLM'])
        with h5py.File(self.source,'r') as f:
            np.testing.assert_array_equal(f['ptychogramx'][()],self.frames)

    def test_axis_mapping(self):
        path,_ = entry.prepare(self.args('--position-order','xy','--flip-y'))
        expected = self.coords.T[:,[1,0]]*1e-3
        expected[:,0] *= -1
        expected -= expected.mean(axis=0)
        with h5py.File(path,'r') as f:
            np.testing.assert_allclose(f['encoder'][()],expected,atol=1e-15)

    def test_reject_overwrite(self):
        a = self.args()
        entry.prepare(a)
        with self.assertRaises(SystemExit):
            self.args()

    def test_reject_negative_intensity(self):
        with h5py.File(self.source,'a') as f:
            f['ptychogramx'][7,20,20] = -1
        with self.assertRaisesRegex(ValueError,'Frame 7'):
            entry.prepare(self.args())

    def test_reject_crop_larger_than_input(self):
        with self.assertRaisesRegex(ValueError,'Crop'):
            entry.prepare(self.args('--detector-crop','64'))

    def test_cpu_engine_and_short_reconstruction(self):
        a = self.args()
        runtime = entry.require_runtime(cpu=True)
        path,config = entry.prepare(a)
        _,recon,params,engine = entry.build_engine(path,a)
        self.assertEqual(params.propagatorType,'Fraunhofer')
        self.assertEqual(params.intensityConstraint,'standard')
        self.assertFalse(params.gpuSwitch)
        for key in entry.DISABLED:
            self.assertFalse(getattr(params,key),key)
        self.assertEqual(recon.Np,32)
        self.assertEqual(len(np.unique(recon.positions,axis=0)),225)
        self.assertEqual(engine.numIterations,2)
        entry.reconstruct(path,a,config,runtime)
        with np.load(a.out/'reconstruction.npz') as result:
            self.assertEqual(result['probe_rec'].shape,(32,32))
            self.assertEqual(len(result['error']),2)
            self.assertTrue(np.isfinite(result['obj_rec']).all())
            self.assertTrue(np.iscomplexobj(result['obj_rec']))
        summary = json.loads((a.out/'summary.json').read_text())
        self.assertEqual(summary['frame_visits'],450)
        self.assertGreater(summary['reconstruct_call_seconds'],0)
        for name in ['reconstruction.png','reconstruction.pdf','error_curve.png','error_curve.pdf']:
            self.assertGreater((a.out/name).stat().st_size,100)


if __name__ == '__main__':
    unittest.main()
