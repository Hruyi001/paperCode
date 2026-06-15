import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_fcfe_modules_keep_feature_shape():
    from models.ConvNext.fcfe_modules import FSM, MGSE, SEM

    rgb = torch.randn(2, 32, 8, 8)
    normal = torch.randn(2, 32, 8, 8)

    mgse = MGSE(32)
    sem = SEM(32)
    fsm = FSM(32)

    x = mgse(rgb, normal)
    assert x.shape == rgb.shape
    x = sem(x)
    assert x.shape == rgb.shape
    x = fsm(x)
    assert x.shape == rgb.shape


def test_fcfe_modules_start_near_identity_for_finetuning():
    from models.ConvNext.fcfe_modules import FSM, MGSE, SEM

    rgb = torch.randn(2, 32, 8, 8)
    normal = torch.randn(2, 32, 8, 8)

    for module, args in [(MGSE, (32,)), (SEM, (32,)), (FSM, (32,))]:
        layer = module(*args)
        output = layer(rgb, normal) if module is MGSE else layer(rgb)
        assert torch.allclose(output, rgb, atol=1e-6)


def test_two_view_net_accepts_rgb_normal_pairs_without_pretrained_download():
    from models.model import two_view_net

    model = two_view_net(
        5,
        block=2,
        M=4,
        return_f=True,
        resnet=False,
        use_fcfe=True,
        use_normals=True,
        pretrained=False,
    )
    model.train()

    rgb = torch.randn(2, 3, 64, 64)
    normal = torch.randn(2, 3, 64, 64)
    y1, y2 = model((rgb, normal), (rgb, normal))

    assert isinstance(y1, tuple)
    assert isinstance(y2, tuple)
    assert len(y1) == 2
    assert len(y2) == 2


def test_normal_cache_maps_rgb_path_to_cached_normal():
    from datasets.omnidata_normals import NormalCache

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "train"
        rgb_path = root / "satellite" / "0001" / "sample.jpg"
        rgb_path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(rgb_path)

        normal_root = Path(tmp) / "normals"
        normal_path = normal_root / "satellite" / "0001" / "sample_normal.png"
        normal_path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(128, 128, 255)).save(normal_path)

        cache = NormalCache(
            normal_dir=str(normal_root),
            data_root=str(root),
            auto_generate=False,
        )
        normal = cache.get(str(rgb_path))

        assert normal.mode == "RGB"
        assert normal.size == (8, 8)
        assert cache.normal_path(str(rgb_path)) == str(normal_path)


def test_normal_cache_regenerates_truncated_cached_normal():
    from datasets.omnidata_normals import NormalCache

    class FakeCache(NormalCache):
        def generate(self, rgb_path, normal_path):
            Image.new("RGB", (8, 8), color=(128, 128, 255)).save(normal_path)

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "test"
        rgb_path = root / "query_drone" / "0001" / "sample.jpg"
        rgb_path.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), color=(10, 20, 30)).save(rgb_path)

        normal_root = Path(tmp) / "normals"
        normal_path = normal_root / "query_drone" / "0001" / "sample_normal.png"
        normal_path.parent.mkdir(parents=True)
        normal_path.write_bytes(b"")

        cache = FakeCache(normal_dir=str(normal_root), data_root=str(root), auto_generate=True)
        normal = cache.get(str(rgb_path))

        assert normal.mode == "RGB"
        assert normal.size == (8, 8)


if __name__ == "__main__":
    test_fcfe_modules_keep_feature_shape()
    test_fcfe_modules_start_near_identity_for_finetuning()
    test_two_view_net_accepts_rgb_normal_pairs_without_pretrained_download()
    test_normal_cache_maps_rgb_path_to_cached_normal()
    test_normal_cache_regenerates_truncated_cached_normal()
    print("fcfe smoke tests passed")
