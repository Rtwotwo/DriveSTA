"""Exercise the training entry with the actual released Hydra recipes."""
from importlib import import_module
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from navsim.planning.script.run_training import main


@pytest.mark.parametrize("profile,runner", [
    ("base_proposal", "proposal_real"), ("base", "refinement_real"),
    ("scale_proposal", "proposal_mixed"), ("scale", "refinement_mixed"),
])
def test_released_recipe_enters_selected_training_runner(monkeypatch, profile, runner):
    root = Path(__import__("navsim").__file__).resolve().parents[1]
    directory = root / "navsim/planning/script/config/training"
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name="drivesta_" + profile)
    implementation = import_module("navsim.planning.training.drivesta.runners." + runner)
    received = []
    monkeypatch.setattr(implementation, "run", received.append)
    main.__wrapped__(cfg)
    assert received == [cfg]
