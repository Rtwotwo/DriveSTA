"""Resolve project and shared NAVSIM resources from one place."""

import importlib.util
import os
from pathlib import Path
import sys


def project_root() -> Path:
    override = os.environ.get("DRIVESTA_ROOT") or os.environ.get("DRIVESCORER_ROOT")
    candidates = (
        [Path(override).expanduser().resolve()] if override else Path(__file__).resolve().parents
    )
    for root in candidates:
        if (root / "navsim").is_dir() and (root / "feedback").is_dir():
            return root
    raise FileNotFoundError("Set DRIVESTA_ROOT to the project containing navsim/ and feedback/.")


def bootstrap_navsim(visualization: bool = False) -> Path:
    root = project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.environ.setdefault("NUPLAN_MAPS_ROOT", str(root / "data/maps"))
    if visualization:
        import navsim

        if importlib.util.find_spec("navsim.visualization") is None:
            # The curated repository reuses the original NAVSIM rendering library.
            override = os.environ.get("NAVSIM_VISUALIZATION_ROOT")
            shared = (
                Path(override).expanduser().resolve()
                if override
                else root.parent / "chainflow/navsim"
            )
            if not (shared / "visualization/bev.py").is_file():
                raise FileNotFoundError(
                    "Set NAVSIM_VISUALIZATION_ROOT to a navsim directory with visualization/bev.py."
                )
            if str(shared) not in navsim.__path__:
                navsim.__path__.append(str(shared))
    return root
