"""Durable scene blocks. Published files, not progress counters, define recovery."""

import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def retry_io(function):
    for attempt, delay in enumerate((1, 5, 15, None)):
        try:
            return function()
        except OSError as e:
            if (
                e.errno not in (errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN)
                or delay is None
            ):
                raise
            time.sleep(delay)


def atomic(path, writer, validator=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")

    def commit():
        with open(temporary, "wb") as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        if validator:
            validator(temporary)
        os.replace(temporary, path)
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    try:
        retry_io(commit)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path, value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("Unsupported JSON object: " + type(item).__name__)

    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=scalar).encode()
    atomic(path, lambda f: f.write(data), lambda p: json.loads(p.read_text()))


def read_json(path):
    return json.loads(Path(path).read_text())


def write_npz(path, **arrays):
    def verify(p):
        with np.load(p, allow_pickle=False) as z:
            for key, value in arrays.items():
                if not np.array_equal(z[key], value, equal_nan=True):
                    raise ValueError("NPZ readback failed: " + key)

    atomic(path, lambda f: np.savez(f, **arrays), verify)


@contextlib.contextmanager
def lock(directory):
    """Kernel-owned lock releases on process death, including SIGKILL."""
    import fcntl

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        f.seek(0)
        f.truncate()
        f.write(json.dumps(dict(host=socket.gethostname(), pid=os.getpid(), started=time.time())))
        f.flush()
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def validate_block(z, expected_keys=None):
    keys, trajectories, labels = z["keys"], z["trajectories"], z["labels"]
    n = len(keys)
    if keys.shape != (n, 2) or trajectories.shape != (n, 8, 3) or labels.shape != (n, 7):
        raise ValueError("Invalid block shape")
    if keys.dtype.kind not in "iu" or len(np.unique(keys, axis=0)) != n:
        raise ValueError("Invalid or duplicate integer goals")
    if not np.isfinite(trajectories).all() or not np.isfinite(labels).all():
        raise ValueError("Nonfinite trajectories/labels")
    if (labels < -1e-8).any() or (labels > 1 + 1e-8).any():
        raise ValueError("Labels outside raw 0..1")
    if expected_keys is not None and not np.array_equal(keys, expected_keys):
        raise ValueError("Committed block differs from immutable phase plan")


def load_blocks(directory):
    paths = sorted(Path(directory).glob("phases/*/block_*.npz"))
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            validate_block(z)
            yield path, {k: z[k] for k in z.files}
