import argparse
from pathlib import Path

from tqdm import tqdm

from datasets.omnidata_normals import NormalCache


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


def iter_images(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main():
    parser = argparse.ArgumentParser(description='Precompute Omnidata normal cache')
    parser.add_argument('--data_root', required=True, type=str)
    parser.add_argument('--normal_dir', required=True, type=str)
    parser.add_argument('--omnidata_repo', default='/root/code/omnidata_models', type=str)
    parser.add_argument('--omnidata_root', default='/root/code/omnidata_models/pretrained_models', type=str)
    parser.add_argument('--device', default='auto', type=str)
    parser.add_argument('--shard_id', default=0, type=int)
    parser.add_argument('--num_shards', default=1, type=int)
    parser.add_argument('--limit', default=0, type=int)
    args = parser.parse_args()

    cache = NormalCache(
        normal_dir=args.normal_dir,
        data_root=args.data_root,
        omnidata_repo=args.omnidata_repo,
        omnidata_root=args.omnidata_root,
        auto_generate=True,
        device=args.device,
    )
    images = [path for i, path in enumerate(iter_images(args.data_root)) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        images = images[:args.limit]
    for path in tqdm(images, desc=f'shard {args.shard_id}/{args.num_shards}'):
        normal_path = Path(cache.normal_path(str(path)))
        if normal_path.exists():
            continue
        cache.generate(str(path), str(normal_path))


if __name__ == '__main__':
    main()
