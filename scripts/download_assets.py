"""Download and verify DriveSTA weights and trajectory archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='*', choices=['base', 'scale', 'base_proposal', 'scale_proposal'], default=[])
    parser.add_argument('--trajectories', choices=['navtrainval', 'simscale', 'all'])
    parser.add_argument('--metadata', action='store_true')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--revision', default='main', help='Pin a Hub commit for reproducibility')
    parser.add_argument('--plan', action='store_true', help='Print selected canonical paths without downloading')
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'docs/release/files.json').read_text())['files']
    selected = []
    for item in inventory:
        name = item['path']
        wanted = name in {f'weights/Checkpoints/{profile}.ckpt' for profile in args.profiles}
        wanted |= bool(args.metadata or args.trajectories) and '/Metadata/' in name
        wanted |= bool(args.trajectories in ('navtrainval', 'all')) and '/Navtrainval/' in name
        wanted |= bool(args.trajectories in ('simscale', 'all')) and '/SimScale/' in name
        if wanted:
            selected.append(item)
    if not selected:
        parser.error('Select --profiles, --metadata or --trajectories')
    if args.plan:
        print(json.dumps(selected, indent=2))
        return
    from huggingface_hub import HfApi, hf_hub_download
    repository = 'Rtwotwo/DriveSTA'
    # Resolve once so a concurrent repository update cannot mix two revisions.
    revision = HfApi().model_info(repository, revision=args.revision).sha
    available = set(HfApi().list_repo_files(repository, revision=revision))
    aliases = json.loads((ROOT / 'navsim/common/compatibility/hub_files.json').read_text())
    for item in selected:
        target = args.destination.resolve() / item['path']
        if target.is_file() and target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']:
            print(f'Verified {item["path"]}', flush=True)
            continue
        remote = item['path'] if item['path'] in available else aliases[item['path']]
        cached = Path(hf_hub_download(repository, filename=remote, revision=revision))
        if cached.stat().st_size != item['bytes'] or sha256(cached) != item['sha256']:
            raise ValueError(f'Download checksum mismatch: {item["path"]}')
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
        try:
            shutil.copyfile(cached, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Saved {item["path"]}', flush=True)
    print(f'Hub revision: {revision}')


if __name__ == '__main__':
    main()

