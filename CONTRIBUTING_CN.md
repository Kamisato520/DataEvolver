# 贡献指南

当前只接受以下范围的改动：

- 数据门禁（readiness gate）
- Blender/T2I 合成编排
- 质检/过滤与 manifest 产出
- 评估型 VLM 训练流程

请优先对齐：

- `skills/dataset-synthesis-gate/`
- `skills/dataset-eval-pipeline/`
- `refine-logs/MIGRATED_SETUP_API_GUIDE.md`

所有文档与改动都应围绕统一交付：`dataset_manifest + 指标规范 + 评估型 VLM`。
