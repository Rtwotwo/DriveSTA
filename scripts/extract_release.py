"""Verify and safely extract independently compressed DriveSTA archives."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    destination = destination.resolve()
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or not member.isfile():
                raise ValueError(f"Unsafe archive member: {member.name}")
            target = destination.joinpath(*name.parts)
            if destination not in target.resolve().parents:
                raise ValueError(f"Archive member escapes destination: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Never follow an existing symlink or silently overwrite local edits.
            with target.open("xb") as output, stream.extractfile(member) as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    output.write(block)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--destination", type=Path, default=Path.cwd())
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=1, help="Independent archives to process concurrently"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/release/archives.json",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if len({path.name for path in args.archives}) != len(args.archives):
        parser.error("Each archive may be supplied only once")
    expected = {
        item["archive"]: item["archive_sha256"]
        for item in json.loads(args.manifest.read_text())["archives"]
    }

    def process(archive):
        if archive.name not in expected or sha256(archive) != expected[archive.name]:
            raise ValueError(f"Archive checksum mismatch or unknown archive: {archive}")
        if not args.verify_only:
            extract(archive, args.destination)
        print(f"{'Verified' if args.verify_only else 'Extracted'} {archive.name}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(process, args.archives))
    if not args.verify_only:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from navsim.common.compatibility.indexes import prepare_indexes
        prepare_indexes(args.destination)


if __name__ == "__main__":
    main()
