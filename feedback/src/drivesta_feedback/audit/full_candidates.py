"""Read-only full-NAVTEST best-of-64 diagnostics for an existing evaluation.

Uses the evaluation's resolved agent config and deterministic token-conditioned
sampling. Scores candidates independently against the official PDM reference.
Does not select checkpoints, train, or change any model implementation.
"""

import os

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CHAINFLOW_ENABLE_RAY"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv, hashlib, json, multiprocessing as mp, time
from pathlib import Path
import numpy as np


def reference_rows(evaluation):
    meta = json.loads((evaluation / "summary.json").read_text())
    assert meta["complete"] and meta["failed"] == 0
    rows = [r for r in csv.DictReader(Path(meta["csv"]).open()) if r["token"] != "average"]
    ref = {r["token"]: r for r in rows}
    assert len(ref) == len(rows) == 12146
    assert all(r["valid"] == "True" and np.isfinite(float(r["score"])) for r in rows)
    return meta, ref


def score_job(job):
    token, path, dest, ref_score, log_name, verify_all = job
    from navsim.evaluate.drivesta.proposal.online import score_scene

    file = Path(dest) / (token + ".npz")
    with np.load(file, allow_pickle=False) as stored:
        data = {k: stored[k] for k in stored.files if k != "labels"}
    trajectories = data["trajectories"]
    digest = hashlib.sha256(trajectories.tobytes() + data["prediction"].tobytes()).hexdigest()
    cache = file.with_suffix(".metrics.json")
    if cache.exists():
        cached = json.loads(cache.read_text())
        if (
            cached["prediction_hash"] == digest
            and cached["metric_cache"] == path
            and cached["reference_score"] == ref_score
            and (not verify_all or cached["verify_all"])
        ):
            return cached["row"]
    labels = score_scene((path, trajectories))
    idx = int(data["prediction"].argmax())
    assert labels.shape == (64, 7) and np.isfinite(labels).all()
    selected, oracle = labels[idx, 6] * 100, labels[:, 6].max() * 100
    # Public singleton parity includes all components, not just the total score.
    from drivesta_feedback.batch.source import Evaluator
    import lzma, pickle

    with lzma.open(path, "rb") as stream:
        evaluator = Evaluator(pickle.load(stream))
    indices = range(64) if verify_all else [idx]
    parity = max(float(np.abs(evaluator.score(trajectories[j]) - labels[j]).max()) for j in indices)
    assert parity < 1e-8, (token, "singleton", parity)
    csv_error = float(selected - float(ref_score) * 100)
    # Record every discrepancy; do not silently drop inconsistent scenes.
    tmp = file.with_name(file.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **data, labels=labels)
    os.replace(tmp, file)
    row = dict(
        token=token,
        log_name=log_name,
        selected_pdms=float(selected),
        oracle_pdms=float(oracle),
        regret=float(oracle - selected),
        zero=bool(selected == 0),
        choice=idx,
        csv_error=csv_error,
        singleton_error=parity,
        high_fraction=float((labels[:, 6] >= 0.95).mean()),
    )
    cached = dict(
        prediction_hash=digest,
        metric_cache=path,
        reference_score=ref_score,
        verify_all=verify_all,
        row=row,
    )
    temp = cache.with_name(cache.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(cached))
    os.replace(temp, cache)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--verify-all", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--resume-inference", action="store_true")
    p.add_argument("--repair-selection", action="store_true")
    args = p.parse_args()
    from drivesta_feedback.paths import project_root

    root = project_root()
    for k, v in dict(
        NAVSIM_EXP_ROOT=root / "exp",
        OPENSCENE_DATA_ROOT=root / "data",
        NUPLAN_MAPS_ROOT=root / "data/maps",
        NUPLAN_MAP_VERSION="nuplan-maps-v1.0",
    ).items():
        os.environ.setdefault(k, str(v))
    meta, ref = reference_rows(args.evaluation)
    if args.aggregate:
        rows = []
        for rank in range(args.world):
            shard = json.loads((args.output / f"rank_{rank:02d}/result.json").read_text())
            assert shard["checkpoint"] == meta["checkpoint"]
            rows.extend(shard["rows"])
        assert len(rows) == len(ref) and len({r["token"] for r in rows}) == len(ref)
        assert {r["token"] for r in rows} == set(ref)
        summary = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("selected_pdms", "oracle_pdms", "regret", "high_fraction")
        }
        summary.update(
            scenes=len(rows),
            candidates=64 * len(rows),
            zero_count=sum(r["zero"] for r in rows),
            zero_percent=100 * np.mean([r["zero"] for r in rows]),
            max_csv_error=max(abs(r["csv_error"]) for r in rows),
            max_singleton_error=max(r["singleton_error"] for r in rows),
        )
        report = dict(
            evaluation=str(args.evaluation),
            checkpoint=meta["checkpoint"],
            precision="FP32",
            candidates_per_scene=64,
            scope="Full NAVTEST, same checkpoint and selected trajectory",
            summary=summary,
        )
        # Never declare parity when the pool does not reproduce reported selections.
        report["selection_parity_passed"] = summary["max_csv_error"] < 1e-5
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with (args.output / "per_scene.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["token"]))
        print(json.dumps(report), flush=True)
        assert report[
            "selection_parity_passed"
        ], "Full result is not eligible for table until discrepancies are resolved"
        return
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataloader import SceneLoader, MetricCacheLoader
    from navsim.planning.training.dataset import Dataset
    from navsim.planning.training.collate import drivesta_collate_fn_for_agent
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load(args.evaluation / "resolved_eval_config.yaml")
    assert str(cfg.agent.checkpoint_path) == meta["checkpoint"]
    assert cfg.agent.checkpoint_strict_load
    assert cfg.trainer.params.precision == "32-true"
    torch.set_num_threads(1)
    pl.seed_everything(int(cfg.seed), workers=True)
    torch.backends.cudnn.benchmark = False
    if cfg.get("deterministic", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
    selected = sorted(ref)[args.rank :: args.world]
    if args.limit:
        selected = selected[: args.limit]
    output = args.output / f"rank_{args.rank:02d}"
    previous = None
    if args.repair_selection:
        assert args.batch_size == 1 and not args.resume_inference
        previous = json.loads((output / "result.json").read_text())
        selected = [r["token"] for r in previous["rows"] if abs(r["csv_error"]) >= 1e-5]
        if not selected:
            print("No selection discrepancies require replay", flush=True)
            return
        backup = output / "result_before_single_replay.json"
        assert not backup.exists()
        backup.write_text(json.dumps(previous, indent=2) + "\n")
        print(json.dumps(dict(phase="single_replay", count=len(selected))), flush=True)
    output.mkdir(parents=True, exist_ok=args.resume_inference or args.repair_selection)
    started = time.time()
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    remaining = [
        t for t in selected if not (args.resume_inference and (output / (t + ".npz")).is_file())
    ]
    sf = instantiate(cfg.train_test_split.scene_filter)
    sf.tokens = remaining
    loader = SceneLoader(
        data_path=Path(cfg.navsim_log_path),
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        scene_filter=sf,
        sensor_config=agent.get_sensor_config(),
    )
    assert set(loader.tokens) == set(remaining)
    ds = Dataset(
        scene_loader=loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=[],
        cache_path=None,
        force_cache_computation=False,
        append_token_to_batch=True,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        collate_fn=drivesta_collate_fn_for_agent(agent),
        pin_memory=True,
    )
    source = root / Path(*str(cfg.agent._target_).split(".")[:-1])
    fingerprint = {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in source.parent.glob("*.py")
    }
    manifest = output / (
        "manifest_replay.json"
        if args.repair_selection
        else "manifest_resume.json" if args.resume_inference else "manifest.json"
    )
    manifest.write_text(
        json.dumps(
            dict(
                checkpoint=meta["checkpoint"],
                batch_size=args.batch_size,
                resumed_count=len(selected) - len(remaining),
                config_sha256=hashlib.sha256(
                    (args.evaluation / "resolved_eval_config.yaml").read_bytes()
                ).hexdigest(),
                source_hashes=fingerprint,
                seed=int(cfg.seed),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32,
                tokens=selected,
            ),
            indent=2,
        )
    )
    for i, (features, targets, tokens) in enumerate(dl, 1):
        features = {
            k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in features.items()
        }
        with torch.inference_mode():
            pred = agent.forward(features, targets)
        for b, token in enumerate(tokens):
            tr = pred["proposals"][b].float().cpu().numpy()
            scores = pred["pdm_score"][b].float().cpu().numpy()
            chosen = pred["trajectory"][b].float().cpu().numpy()
            assert (
                tr.shape == (64, 8, 3)
                and scores.shape == (64,)
                and np.isfinite(tr).all()
                and np.isfinite(scores).all()
            )
            assert np.array_equal(chosen, tr[scores.argmax()])
            np.savez_compressed(output / (token + ".npz"), trajectories=tr, prediction=scores)
        if i % 25 == 0 or i == len(dl):
            print(
                json.dumps(
                    dict(
                        phase="inference",
                        done=min(i * args.batch_size, len(ds)) + len(selected) - len(remaining),
                        total=len(selected),
                        seconds=time.time() - started,
                    )
                ),
                flush=True,
            )
    del agent, dl, ds, loader, pred, features
    torch.cuda.empty_cache()
    caches = MetricCacheLoader(Path(cfg.metric_cache_path))
    jobs = [
        (
            t,
            str(caches.metric_cache_paths[t]),
            str(output),
            ref[t]["score"],
            ref[t]["log_name"],
            args.verify_all,
        )
        for t in selected
    ]
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        for i, row in enumerate(pool.map(score_job, jobs), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(jobs):
                progress = dict(
                    phase="scoring",
                    done=i,
                    total=len(jobs),
                    seconds=time.time() - started,
                    max_csv_error=max(abs(r["csv_error"]) for r in rows),
                )
                (output / "progress.json").write_text(json.dumps(progress))
                print(json.dumps(progress), flush=True)
    if previous is not None:
        updated = {r["token"]: r for r in rows}
        rows = [updated.get(r["token"], r) for r in previous["rows"]]
    result = dict(checkpoint=meta["checkpoint"], seconds=time.time() - started, rows=rows)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                phase="complete",
                scenes=len(rows),
                seconds=result["seconds"],
                max_csv_error=max(abs(r["csv_error"]) for r in rows),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
