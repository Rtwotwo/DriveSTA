# HUGSIM trajectory videos

The project page shows the same 51 selected closed-loop episodes and the same
321.8-second montage. Orange ribbons and white-edged dots visualize the selected
DriveSTA-Scale Refinement prediction recorded at each observation: eight future
waypoints, sampled every 0.5 seconds over a 4-second horizon. Videos retain the
original 4 FPS timing; the montage repeats frames at 30 FPS.

The overlay uses the original `adapter_frames.jsonl` predictions, paired with
`infos.pkl` by frame index and timestamp. It uses the simulator's camera
intrinsics, `cam_rect`, relative camera poses, and `ground_param.pkl` road-height
reference. It does not infer a trajectory from the subsequently executed path.
The aggregate scorer's `data.pkl` representation is not used as an overlay
source: its trajectory conversion omits the eighth point and uses post-step
poses, whereas the observation video contains pre-step frames.

All eight predicted waypoints are projected. Markers retain their original
coordinates and times. Points outside the camera field of view are clipped;
there are no bird's-eye insets, extrapolated points or relocated markers.
Close points can be outside the image during low-speed driving, and distant
points can overlap in perspective. Dense samples only tessellate the orange
ribbon, whose width is a visualization aid, not a predicted vehicle envelope.

## Reproduce

Dependencies: Python, NumPy, SciPy, PyYAML, OpenCV, and FFmpeg/FFprobe. Use only
trusted simulator pickle files. `EPISODES` contains
`<dataset>/<scene>_<mode>/`; `MODELS` contains
`<dataset>/<scene>/ground_param.pkl`. Those original evaluation records and
licensed scene assets are separate from the website assets.

```bash
python tools/render_hugsim_trajectories.py \
  --repo . --episodes "$EPISODES" --models "$MODELS" \
  --output "$WORK/annotated" --workers 4

python tools/update_hugsim_demo.py \
  --repo . --annotated-videos "$WORK/annotated/videos" \
  --work "$WORK/montage"
```

Add `--samples-only` to the first command to inspect four frames per episode
before rendering. Review the generated frames and `trajectory-audit.json`, then
copy the videos and posters to the matching paths in the existing selection
manifest. The montage updater preserves the original intro/NAVSIM duration and
reuses `docs/assets/data/demo-edit.json` for all HUGSIM scene order, lane positions
and trimming. Its intro/NAVSIM frames are re-encoded for codec compatibility.

The published validation summary is
`docs/assets/data/hugsim-trajectory-render.json`. It records frame counts,
trace SHA-256 digests, and visibility counts across the three front cameras.
Visibility counts refer to projected waypoint centers with a four-pixel margin,
not to whether a waypoint is occluded by a rendered object.

Before deployment, run:

```bash
python tools/test_hugsim_projection.py
python tools/build_site.py --output "$WORK/site"
python tools/check_site.py --site-root "$WORK/site"
```
