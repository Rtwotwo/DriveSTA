# NAVSIM v2 评分依赖

从用户指定的 `chainflow/clones/navsim_v2_eval` 复制评分需要的依赖闭包。
这是 EPDMS 仿真库，不包含训练流程、模型权重或其它 agent 实验分支。

仅改变 Python 命名空间及可信本地 pickle 的类加载路径，评分公式保持来源实现。
完整来源记录见 `source_manifest.json`，原许可证见 `UPSTREAM_LICENSE`。

测评入口位于 `navsim/evaluate/epdms/`，参数位于
`navsim/planning/script/config/epdms/`，使用说明见项目根目录 `docs/EPDMS.md`。
