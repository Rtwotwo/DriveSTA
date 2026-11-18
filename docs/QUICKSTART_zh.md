# DriveSTA 中文快速开始

本发布包含统一的 `DriveSTAAgent` / `DriveSTAModel`、四套配置、训练和测评入口、轨迹挖掘工具及完整教程。Base/Scale 表示训练数据规模；`proposal` 表示仅提案生成与评分，`base` / `scale` 则包含时序精修。模型不按阶段编号拆分目录。

## 1. 安装

进入仓库目录，创建 Python 3.9 环境，按 [INSTALL.md](INSTALL.md) 安装固定依赖及三个本地包。服务器发布目录为 `/data/xinyao/release/DriveSTA`。每次打开终端后设置外部数据目录，并执行：

```bash
export OPENSCENE_DATA_ROOT=/你的/openscene目录
export NAVSIM_CACHE_ROOT=/你的/navsim_cache目录
export NUPLAN_MAPS_ROOT=/你的/maps目录
source scripts/environment.sh
```

数据根目录下应包含 `navsim_logs/` 和 `sensor_blobs/`。轨迹库不包含原始图像、日志、地图和评分缓存。

## 2. 推理

```bash
python scripts/download_assets.py --profiles base
python -m navsim.planning.script.run_inference \
  --profile base --token 场景TOKEN --split test --device cuda \
  --output outputs/prediction.json
```

将 `场景TOKEN` 换成真实测试日志中的 token。输出为自车坐标系下的 8 个轨迹点，每点包含 x、y 和航向，间隔 0.5 秒。服务器已有权重时无需再次下载。完整权重推理无需单独初始化预训练骨干。

## 3. 训练

先按 [DATA_PREPARATION.md](DATA_PREPARATION.md) 下载、解压并检查监督数据、索引、特征缓存和在线评分缓存，再执行：

```bash
bash scripts/training/train_drivesta.sh base_proposal --cfg job --resolve
bash scripts/training/train_drivesta.sh base_proposal preflight_only=true
bash scripts/training/train_drivesta.sh base_proposal
bash scripts/training/train_drivesta.sh base \
  agent.checkpoint_path=/提案模型的/checkpoint.ckpt
```

Scale 对应 `scale_proposal` 和 `scale`，需准备固定混合数据索引及适配后的合成评分缓存。完整轨迹集合有 308,119 个场景；固定 Scale 训练配方使用 232,672 个场景，两者不可混淆。初始化权重和断点续训使用不同参数，详见 [TRAINING.md](TRAINING.md)。

## 4. 测评

```bash
GPUS_PER_NODE=1 bash scripts/evaluation/run_drivesta_pdm_score_evaluation.sh base
```

正式 NAVTEST 测评默认检查 12,146 个场景及完整评分，需预先准备正确版本的 metric cache。单场景推理不等于测评；少量场景的冒烟检查结果也不能当作完整 benchmark 分数。命令、输出和单场景评分检查见 [INFERENCE.md](INFERENCE.md)。

## 5. 修改参数的位置

模型参数：`navsim/planning/script/config/common/agent/`；训练配方：`config/training/`；测评参数：`config/pdm_scoring/`；发布权重参数快照：`config/checkpoints/`。脚本只负责启动和运行时路径，模型行为由 YAML 配置控制。
