"""Exercise public launchers without starting a training or evaluation job."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("base_proposal", "base", "scale_proposal", "scale")


def launch(tmp_path, script, variant, overrides=(), environment=None):
    root = tmp_path / "release with spaces"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in (script, "environment.sh"):
        (scripts / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / name, scripts / name)
    capture = tmp_path / "argv.json"
    recorder = tmp_path / "python_recorder"
    recorder.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['CAPTURE'], 'w') as stream:\n"
        "    json.dump({'argv':sys.argv[1:],'env':dict(os.environ)}, stream)\n"
    )
    recorder.chmod(0o755)
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "DRIVESTA_ROOT",
            "NAVSIM_EXP_ROOT",
            "CHECKPOINT_PATH",
            "OUTPUT_DIR",
            "METRIC_CACHE_PATH",
            "GPUS_PER_NODE",
            "RANK_WEIGHT",
            "OPENSCENE_DATA_ROOT",
            "NAVSIM_CACHE_ROOT",
        }
    }
    env.update(PYTHON_BIN=str(recorder), CAPTURE=str(capture))
    env.update(environment or {})
    result = subprocess.run(
        ["bash", str(scripts / script), variant, *overrides],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    return result, json.loads(capture.read_text()) if capture.exists() else None, root


@pytest.mark.parametrize("variant", VARIANTS)
def test_train_selects_release_recipe_and_preserves_overrides(tmp_path, variant):
    result, captured, root = launch(tmp_path, "training/train_drivesta.sh", variant, ["preflight_only=true"])
    assert result.returncode == 0, result.stderr
    assert "--config-name=drivesta_" + variant in captured["argv"]
    assert captured["argv"][-1] == "preflight_only=true"
    assert captured["env"]["DRIVESTA_ROOT"] == str(root)


@pytest.mark.parametrize("variant", VARIANTS)
def test_evaluate_preserves_checkpoint_config_and_fp32_protocol(tmp_path, variant):
    checkpoint = tmp_path / "epoch=12 weights.ckpt"
    result, captured, root = launch(
        tmp_path,
        "evaluation/run_drivesta_pdm_score_evaluation.sh",
        variant,
        ["expected_eval_scenes=2"],
        {"CHECKPOINT_PATH": str(checkpoint), "GPUS_PER_NODE": "2"},
    )
    assert result.returncode == 0, result.stderr
    args = captured["argv"]
    assert "--nproc-per-node=2" in args
    assert "trainer.params.precision=32-true" in args
    assert "agent.checkpoint_path='" + str(checkpoint) + "'" in args
    assert (
        "training_config_path='" + str(root / "navsim/planning/script/config/checkpoints" / (variant + ".yaml")) + "'" in args
    )
    assert args[-1] == "expected_eval_scenes=2"


@pytest.mark.parametrize("script", ["training/train_drivesta.sh", "evaluation/run_drivesta_pdm_score_evaluation.sh"])
def test_unknown_variant_fails_before_python(tmp_path, script):
    result, captured, _ = launch(tmp_path, script, "unknown")
    assert result.returncode == 2
    assert captured is None


def test_external_dataset_and_cache_roots_are_preserved(tmp_path):
    env = {"OPENSCENE_DATA_ROOT": "/datasets/external", "NAVSIM_CACHE_ROOT": "/caches/external"}
    result, captured, _ = launch(tmp_path, "training/train_drivesta.sh", "scale_proposal", environment=env)
    assert result.returncode == 0
    for key, value in env.items():
        assert captured["env"][key] == value


@pytest.mark.parametrize("variant", VARIANTS)
def test_default_evaluation_checkpoint_uses_weights_directory(tmp_path, variant):
    result, captured, root = launch(tmp_path, "evaluation/run_drivesta_pdm_score_evaluation.sh", variant)
    assert result.returncode == 0, result.stderr
    checkpoint = root / "weights" / "Checkpoints" / (variant + ".ckpt")
    assert "agent.checkpoint_path='" + str(checkpoint) + "'" in captured["argv"]
