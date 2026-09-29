import unittest
from dataclasses import replace
import torch
from multiwave.config import Config
from multiwave.scene import simulate
from multiwave.models import SharedAmplitudeModel, amplitude_tv
from multiwave.probes import PixelProbes
from multiwave.physics import amplitude_loss
from multiwave.reconstruct import backward_shared_data


class DetectorTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_larger_detector_nested_crop_and_input(self):
        cfg = replace(Config.preset('smoke'), pad_factor=4)
        a, b = simulate(cfg), simulate(replace(cfg, detector_size=128))
        torch.testing.assert_close(a.objects,b.objects,rtol=0,atol=0)
        torch.testing.assert_close(a.operator.probes,b.operator.probes,rtol=0,atol=0)
        torch.testing.assert_close(a.clean,b.clean[:,48:80,48:80],rtol=0,atol=0)
        self.assertEqual(tuple(b.input_stack.shape), (1,20,64,64))
        self.assertEqual(tuple(b.measured.shape), (25,128,128))

    def test_chunk_vjp_matches_network_and_probe_gradients(self):
        cfg = replace(Config.preset('smoke'), detector_size=128,pad_factor=4,chunk=3,tv_weight=.001)
        scene = simulate(cfg)
        for activation in ('sigmoid','softplus'):
            cfg = replace(cfg, unet_activation=activation)
            net = SharedAmplitudeModel(cfg,'unet_shared_amp',scene.input_stack)
            probe = PixelProbes(cfg)
            a,_,_ = net(); p = probe()
            loss = amplitude_loss(scene.operator(a,scene.train,probes=p),scene.measured[scene.train])
            loss = loss+cfg.tv_weight*amplitude_tv(a.abs(),scene.roi)
            loss.backward()
            params = list(net.parameters())+list(probe.parameters())
            reference = [v.grad.clone() for v in params]
            net.zero_grad();probe.zero_grad()
            a,_,_ = net();p = probe()
            backward_shared_data(a,p,scene,cfg)
            for v,g in zip(params,reference):
                torch.testing.assert_close(v.grad,g,rtol=2e-4,atol=2e-7)

    def test_resolved_preset_and_matching_initialization(self):
        cfg = Config.preset('resolved').validate()
        self.assertEqual(cfg.detector_pixels,768)
        s = simulate(replace(Config.preset('smoke'),pixel_parameterization='direct',unet_activation='softplus'))
        c = replace(Config.preset('smoke'),pixel_parameterization='direct',unet_activation='softplus')
        a = SharedAmplitudeModel(c,'pixel_shared_amp',s.input_stack)()[0]
        b = SharedAmplitudeModel(c,'unet_shared_amp',s.input_stack)()[0]
        torch.testing.assert_close(a,b)
        with self.assertRaises(ValueError):
            replace(c,detector_size=200).validate()


if __name__ == '__main__': unittest.main()
