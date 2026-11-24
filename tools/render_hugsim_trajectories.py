"""Project the eight recorded DriveSTA waypoints onto HUGSIM camera videos.

Inputs are the original adapter_frames.jsonl, infos.pkl, simulation config,
ground_param.pkl and video.mp4. No re-inference, future ego positions, waypoint
resampling, time shift, or off-screen waypoint relocation is used. Run only on
trusted HUGSIM pickle files. Dependencies: numpy, scipy, PyYAML, OpenCV, ffmpeg.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

CAMERAS = ('CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT')
ORANGE = (70, 169, 246)  # BGR; warm orange, matching the paper's HUGSIM plans.


def pose(euler, translation, degrees=False):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler('XYZ', euler, degrees=degrees).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def dense_poses(poses):
    # Same five subdivision passes as HUGSIM dense_cam_poses().
    for _ in range(5):
        result = []
        for first, second in zip(poses[:-1], poses[1:]):
            result.append(first)
            if np.linalg.norm(first[:3, 3] - second[:3, 3]) > .1:
                angles = (Rotation.from_matrix(first[:3, :3]).as_euler('XYZ') +
                          Rotation.from_matrix(second[:3, :3]).as_euler('XYZ')) / 2
                result.append(pose(angles, (first[:3, 3] + second[:3, 3]) / 2))
        result.append(poses[-1])
        poses = np.asarray(result)
    return poses


class Projection:
    def __init__(self, config, info, ground_file):
        with ground_file.open('rb') as stream:
            ground_poses, self.camera_height, _ = pickle.load(stream)
        self.ground = dense_poses(ground_poses)
        self.tree = cKDTree(self.ground[:, [0, 2], 3])
        rect = config['camera'].get('cam_rect', {})
        rect = pose(rect.get('rot', [0, 0, 0]), rect.get('trans', [0, 0, 0]), True)
        self.cameras = []
        params = info['cam_params']
        for name in CAMERAS:
            camera = params[name]
            intrinsic = camera['intrinsic']
            width, height = intrinsic['W'], intrinsic['H']
            matrix = np.array([[width / (2 * np.tan(intrinsic['fovx'] / 2)), 0, intrinsic['cx']],
                               [0, height / (2 * np.tan(intrinsic['fovy'] / 2)), intrinsic['cy']],
                               [0, 0, 1]])
            # Exactly _get_obs(): c2w = ego @ v2front @ inv(v2c) @ cam_rect.
            c2ego = params['CAM_FRONT']['v2c'] @ np.linalg.inv(camera['v2c']) @ rect
            self.cameras.append((name, matrix, c2ego, width, height))

    def world(self, xy, info):
        # NAVSIM prediction: x forward, y left. HUGSIM ego: x right, y down, z forward.
        local = np.c_[-xy[:, 1], np.zeros(len(xy)), xy[:, 0], np.ones(len(xy))]
        ego = pose(info['ego_rot'], info['ego_pos'])
        world = (ego @ local.T).T
        _, indices = self.tree.query(world[:, [0, 2]])
        ground = self.ground[indices]
        # HUGSIM ground_height() gives the camera-height reference plane.
        # Project to that plane, then add ground_param's physical camera height.
        temp = world[:, :3].copy()
        temp[:, 1] = 0
        relative = np.einsum('nji,nj->ni', ground[:, :3, :3], temp - ground[:, :3, 3])
        relative[:, 1] = self.camera_height
        world[:, 1] = (np.einsum('nij,nj->ni', ground[:, :3, :3], relative) + ground[:, :3, 3])[:, 1]
        return world

    def project(self, xy, info, camera):
        _, intrinsic, c2ego, _, _ = camera
        world = self.world(xy, info)
        w2c = np.linalg.inv(pose(info['ego_rot'], info['ego_pos']) @ c2ego)
        coordinates = (w2c @ world.T).T[:, :3]
        return coordinates, intrinsic


def image_points(camera_points, intrinsic):
    points = camera_points @ intrinsic.T
    return points[:, :2] / points[:, 2:3]


def near_clip(polygon, near=.10):
    output = []
    for previous, current in zip(np.roll(polygon, 1, axis=0), polygon):
        pin, cin = previous[2] >= near, current[2] >= near
        if pin != cin:
            output.append(previous + (current - previous) * ((near - previous[2]) / (current[2] - previous[2])))
        if cin:
            output.append(current)
    return np.asarray(output)


def draw_plan(frame, trajectory, info, projector):
    xy = trajectory[:, :2]
    # Dense points only tessellate the road ribbon; markers stay at the 8 originals.
    path = np.concatenate([a[None] + np.linspace(0, 1, 12, endpoint=False)[:, None] * (b - a)
                           for a, b in zip(xy[:-1], xy[1:])] + [xy[-1:]])
    tangent = np.gradient(path, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = np.divide(tangent, norm, out=np.tile([1., 0.], (len(path), 1)), where=norm > 1e-8)
    lateral = np.c_[-tangent[:, 1], tangent[:, 0]] * .40
    seen = np.zeros(8, dtype=bool)
    x_offset = 0
    for camera in projector.cameras:
        _, intrinsic, _, width, height = camera
        view = frame[:height, x_offset:x_offset + width]
        left, _ = projector.project(path + lateral, info, camera)
        right, _ = projector.project(path - lateral, info, camera)
        mask = np.zeros((height, width), np.uint8)
        for i in range(len(path) - 1):
            clipped = near_clip(np.array([left[i], left[i+1], right[i+1], right[i]]))
            if len(clipped) >= 3:
                polygon = np.rint(image_points(clipped, intrinsic)).astype(np.int32)
                cv2.fillConvexPoly(mask, polygon, 255, cv2.LINE_AA)
        alpha = mask[..., None].astype(float) / 255 * .57
        view[:] = np.rint(view * (1 - alpha) + np.asarray(ORANGE) * alpha).astype(np.uint8)
        coords, _ = projector.project(xy, info, camera)
        with np.errstate(divide='ignore', invalid='ignore'):
            pixels = image_points(coords, intrinsic)
        valid = (coords[:, 2] >= .10) & np.isfinite(pixels).all(axis=1)
        visible = valid & (pixels[:, 0] >= 4) & (pixels[:, 0] < width - 4) & (pixels[:, 1] >= 4) & (pixels[:, 1] < height - 4)
        seen |= visible
        for a, b in zip(coords[:-1], coords[1:]):
            if max(a[2], b[2]) < .10:
                continue
            a, b = a.copy(), b.copy()
            if a[2] < .10:
                a = b + (a-b) * ((.10-b[2])/(a[2]-b[2]))
            if b[2] < .10:
                b = a + (b-a) * ((.10-a[2])/(b[2]-a[2]))
            endpoints = np.rint(image_points(np.array([a,b]), intrinsic)).astype(int)
            ok, start, end = cv2.clipLine((0, 0, width, height), tuple(endpoints[0]), tuple(endpoints[1]))
            if ok:
                cv2.line(view, start, end, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.line(view, start, end, ORANGE, 1, cv2.LINE_AA)
        for index in np.flatnonzero(visible):
            point = tuple(np.rint(pixels[index]).astype(int))
            neighbors = [j for j in (index-1,index+1) if 0 <= j < 8 and visible[j]]
            separation = min([np.linalg.norm(pixels[index]-pixels[j]) for j in neighbors] or [12.])
            radius = int(np.clip(separation*.38,2,5))
            cv2.circle(view, point, radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(view, point, max(1,radius-2), ORANGE, -1, cv2.LINE_AA)
        x_offset += width
    return frame[:450], seen


def render(item, args, samples_only=False):
    cv2.setNumThreads(1)
    name = f"{item['dataset']}-{item['scene']}-{item['mode']}"
    source = args.episodes / item['dataset'] / f"{item['scene']}_{item['mode']}"
    records = [json.loads(line) for line in (source/'adapter_frames.jsonl').read_text().splitlines()]
    with (source/'infos.pkl').open('rb') as stream:
        infos = pickle.load(stream)
    config = yaml.safe_load((source/'effective_sim_config.yaml').read_text())
    projector = Projection(config, infos[0], args.models/item['dataset']/item['scene']/'ground_param.pkl')
    cap = cv2.VideoCapture(str(source/'video.mp4'))
    assert cap.isOpened(), source
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(infos) == len(records), source
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 4) < 1e-6
    assert abs(len(records) / 4 - item['duration_s']) < 1e-6
    destination = args.output/'videos'/f'{name}.mp4'
    report = dict(dataset=item['dataset'], scene=item['scene'], mode=item['mode'], frames=len(records),
                  visible_waypoint_histogram=[0]*9, source_trace_sha256=hashlib.sha256((source/'adapter_frames.jsonl').read_bytes()).hexdigest())
    targets = set([0, len(records)//4, len(records)//2, 3*len(records)//4])
    process = None
    best_poster, best_key = None, (-1, -float('inf'))
    if not samples_only:
        process = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24',
            '-s','1920x360','-r','4','-i','-','-an','-c:v','libx264','-preset','fast','-crf','19',
            '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(destination)], stdin=subprocess.PIPE)
    try:
        for index, (record, info) in enumerate(zip(records, infos)):
            success, frame = cap.read()
            assert success and frame.shape == (900,2400,3), (name,index)
            assert record['frame'] == index
            assert abs(record['timestamp'] - info['timestamp']) < 1e-7
            assert abs(record['timestamp'] - index/4) < 1e-7
            trajectory = np.asarray(record['trajectory'])
            assert trajectory.shape == (8,3) and np.isfinite(trajectory).all()
            if samples_only and index not in targets:
                continue
            annotated, visible = draw_plan(frame, trajectory, info, projector)
            report['visible_waypoint_histogram'][int(visible.sum())] += 1
            output = cv2.resize(annotated, (1920,360), interpolation=cv2.INTER_AREA)
            if process:
                process.stdin.write(output.tobytes())
            if index in targets:
                cv2.imwrite(str(args.output/'samples'/f'{name}-{index:04d}.jpg'), output, [cv2.IMWRITE_JPEG_QUALITY,95])
            key = (int(visible.sum()), -abs(index-len(records)/2))
            if key > best_key:
                best_key, best_poster = key, output.copy()
                report['poster_frame_index'] = index
                report['poster_visible_waypoints'] = int(visible.sum())
    finally:
        cap.release()
        if process:
            process.stdin.close()
            assert process.wait() == 0, name
    cv2.imwrite(str(args.output/'posters'/f'{name}.jpg'), best_poster, [cv2.IMWRITE_JPEG_QUALITY,92])
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    for sub in ('videos','posters','samples'):
        (args.output/sub).mkdir(parents=True,exist_ok=True)
    items = json.loads((args.repo/'docs/assets/hugsim-selection.json').read_text())
    if args.limit:
        items = items[:args.limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = list(pool.map(lambda item: render(item,args,args.samples_only),items))
    report = dict(episodes=reports, points_per_plan=8, waypoint_dt_s=.5, horizon_s=4,
                  video_fps=4, offscreen_policy='Clip at the calibrated image boundary; do not relocate points or add insets.',
                  provenance='Original per-frame model predictions and matching pre-step camera states.')
    (args.output/'trajectory-audit.json').write_text(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
