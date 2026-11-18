# Install DriveSTA

Use Linux, Python 3.9 and a CUDA-capable PyTorch environment for training and benchmark inference. The pinned stack is PyTorch 2.5.1, torchvision 0.20.1, Lightning 2.4.0, Hydra 1.3.2 and NumPy 1.26.4. The release was validated with the CUDA 12.4 PyTorch build. GPU memory requirements depend on profile, batch size and image resolution; the refined model is substantially larger than the proposal-only model.

## Create the environment

```bash
git clone https://github.com/Rtwotwo/DriveSTA.git
cd DriveSTA
conda create -n drivesta python=3.9 -y
conda activate drivesta
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e nuplan-devkit --no-deps
python -m pip install -e feedback --no-deps
python -m pip install -e . --no-deps
```

For a release archive or an existing server checkout, start with `cd /path/to/DriveSTA` instead of cloning. Use a GPU driver compatible with your chosen PyTorch CUDA build. CUDA Toolkit compilation is not required for the supplied DriveSTA model path.

## Configure paths

```bash
export OPENSCENE_DATA_ROOT=/path/to/openscene
export NUPLAN_MAPS_ROOT=/path/to/maps
export NAVSIM_CACHE_ROOT=/path/to/navsim_cache
export OFFICIAL_PDM_CACHE=/path/to/official_metric_cache_v1
export NAVSIM_EXP_ROOT=/path/to/experiments
export PYTHON_BIN="$(command -v python)"
source scripts/environment.sh
```

The environment script computes `DRIVESTA_ROOT` and `NAVSIM_DEVKIT_ROOT` from its own location and preserves external data/cache/output variables. Source it in each new shell before direct `python -m` commands; public shell launchers source it automatically. Do not point the data root at the `navsim_logs` directory itself: it must be the parent of both `navsim_logs/` and `sensor_blobs/`.

## Check the installation

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from navsim.agents.drivesta.drivesta_agent import DriveSTAAgent; print(DriveSTAAgent.__name__)"
bash scripts/training/train_drivesta.sh base --cfg job --resolve
python -m navsim.planning.script.run_inference --help
```

The configuration check does not load data or start optimization. Continue with [data preparation](DATA_PREPARATION.md), then [inference](INFERENCE.md) or [training](TRAINING.md).

For development checks:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The release bundles its required NAVSIM/nuPlan source. Do not install a second NAVSIM checkout earlier on `PYTHONPATH`, because that can silently resolve a different model or configuration package.
