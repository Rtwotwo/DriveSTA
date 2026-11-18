"""Run DriveSTA on one scene using its current and historical observations."""
import argparse
import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.common.release_paths import project_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('base', 'scale', 'base_proposal', 'scale_proposal'), default='base')
    parser.add_argument('--token', required=True, help='Scene token from the selected driving logs')
    parser.add_argument('--split', default='test', help='Subdirectory of navsim_logs and sensor_blobs')
    parser.add_argument('--log-name', help='Optional log basename to avoid scanning unrelated logs')
    parser.add_argument('--data-root', type=Path, help='Parent of navsim_logs and sensor_blobs')
    parser.add_argument('--checkpoint', type=Path, help='Override the released checkpoint')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=Path('outputs/prediction.json'))
    args = parser.parse_args()

    import os
    root = project_root()
    os.environ.setdefault('DRIVESTA_ROOT', str(root))
    data_root = args.data_root or Path(os.environ.get('OPENSCENE_DATA_ROOT', root / 'data'))
    cfg = OmegaConf.load(root / 'navsim/planning/script/config/checkpoints' / (args.profile + '.yaml'))
    if args.checkpoint:
        cfg.agent.checkpoint_path = str(args.checkpoint.resolve())
    cfg.agent.checkpoint_strict_load = True
    torch.manual_seed(0)
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.to(args.device).eval()
    loader = SceneLoader(
        sensor_blobs_path=data_root / 'sensor_blobs' / args.split,
        data_path=data_root / 'navsim_logs' / args.split,
        scene_filter=SceneFilter(
            num_history_frames=4, num_future_frames=0, frame_interval=1,
            has_route=True, max_scenes=1, tokens=[args.token],
            log_names=[args.log_name] if args.log_name else None,
        ),
        sensor_config=agent.get_sensor_config(),
    )
    if args.token not in loader.tokens:
        raise ValueError(f'Scene {args.token} was not found in {data_root}/navsim_logs/{args.split}')
    agent_input = loader.get_agent_input_from_token(args.token)
    with torch.inference_mode():
        trajectory = agent.compute_trajectory(agent_input)
    poses = trajectory.poses
    if poses.shape != (8, 3) or not torch.isfinite(torch.as_tensor(poses)).all():
        raise RuntimeError(f'Invalid DriveSTA trajectory: shape={poses.shape}')
    result = dict(model='DriveSTA', profile=args.profile, token=args.token,
                  coordinate_frame='ego', columns=['x_m', 'y_m', 'heading_rad'],
                  interval_seconds=0.5, horizon_seconds=4.0, trajectory=poses.tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Saved {len(poses)} future poses to {args.output}')


if __name__ == '__main__':
    main()

