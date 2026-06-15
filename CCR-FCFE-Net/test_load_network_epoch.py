import os
import tempfile
from types import SimpleNamespace

import torch
import yaml

import utils


class DummyModel:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, state):
        self.loaded = state


def _write_opts(path):
    config = {
        'train_all': False,
        'droprate': 0.5,
        'color_jitter': False,
        'batchsize': 8,
        'h': 384,
        'w': 384,
        'share': True,
        'erasing_p': 0.0,
        'lr': 0.01,
        'nclasses': 10,
        'fp16': False,
        'views': 2,
        'block': 2,
        'M': 4,
        'resnet': False,
        'use_fcfe': False,
        'use_normals': False,
        'fcfe_dual_backbone': False,
        'pretrained': False,
    }
    with open(path, 'w') as f:
        yaml.safe_dump(config, f)


def test_load_network_uses_requested_epoch_instead_of_latest():
    with tempfile.TemporaryDirectory() as root:
        model_dir = os.path.join(root, 'exp')
        os.makedirs(model_dir)
        _write_opts(os.path.join(model_dir, 'opts.yaml'))
        open(os.path.join(model_dir, 'net_004.pth'), 'wb').close()
        open(os.path.join(model_dir, 'net_019.pth'), 'wb').close()

        loaded_paths = []
        old_two_view_net = utils.two_view_net
        old_torch_load = torch.load
        try:
            utils.two_view_net = lambda *args, **kwargs: DummyModel()
            torch.load = lambda path: loaded_paths.append(path) or {}
            opt = SimpleNamespace(save_dir=root, which_epoch='004')

            _, _, epoch = utils.load_network('exp', opt)
        finally:
            utils.two_view_net = old_two_view_net
            torch.load = old_torch_load

        assert epoch == '004'
        assert loaded_paths == [os.path.join(root, 'exp', 'net_004.pth')]


if __name__ == '__main__':
    test_load_network_uses_requested_epoch_instead_of_latest()
    print('load_network epoch test passed')
