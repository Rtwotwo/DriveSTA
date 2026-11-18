"""Stable DriveSTA aliases for byte-identical archived training indexes."""
from pathlib import Path
import os

INDEX_FILES = {'proposals.json': 'data/indexes/research/chainflow_feedback_v2/targets32/index.json', 'mixed.json': 'data/indexes/research/chainflow_feedback_v2_scaled237k/data_index.json', 'history.json': 'data/indexes/research/chainflow_feedback_stage2_v2/history_index.json', 'future.json': 'data/indexes/research/chainflow_feedback_stage2_v4/video8_full14_index.json', 'synthetic_video.json': 'data/indexes/research/chainflow_feedback_stage2_v4_scaled237k/simscale_video8_index.json'}

def index_path(path):
    path = Path(path)
    if path.is_file():
        return path
    if path.parent.name == "drivesta" and path.name in INDEX_FILES:
        root = Path(os.environ.get("DRIVESTA_ROOT", Path(__file__).resolve().parents[3]))
        indexes = os.environ.get("DRIVESTA_INDEX_ROOT")
        return (Path(indexes) / Path(INDEX_FILES[path.name]).relative_to("data/indexes/research")) if indexes else root / INDEX_FILES[path.name]
    return path

def prepare_indexes(root):
    """Expose canonical aliases after extraction, preserving all index bytes."""
    import shutil
    root = Path(root).resolve()
    for name, relative in INDEX_FILES.items():
        source = root / relative
        if not source.is_file():
            continue
        target = root / "data/indexes/drivesta" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"Conflicting index alias: {target}")
        else:
            shutil.copy2(source, target)
