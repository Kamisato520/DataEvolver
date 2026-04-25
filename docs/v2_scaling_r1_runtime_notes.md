# v2 Scaling R1 — 运行时记录（故障/质量/路径/分工）

> 从 CLAUDE.md 抽出，集中保存 R1 轮次的细节。CLAUDE.md 中只保留主线和当前进展，详细信息查阅本文件。

## Stage 3 故障记录（2026-04-21）

3 GPU 并行运行 Hunyuan3D `--i23d-devices cuda:0,cuda:1,cuda:2` 时，GPU 1/2 的 Paint 管线（texture baking）全部失败，静默 fallback 到 shape-only GLB。GPU 0 正常。

**根因**：Paint 依赖 CUDA extension（custom_rasterizer / DifferentiableRenderer），可能只在 GPU 0 上编译兼容。Stage 3 的 `try/except` 把 Paint 失败静默吃掉了。

**修复**：在 GPU 0 上重跑 13 个物体的 Stage 3，全部成功。meshes_raw → meshes 已同步。

**教训**：Stage 3 多 GPU 并行需要先验证每个 GPU 的 Paint 是否正常（跑 1 个物体 smoke test）。

## VLM Bootstrap 质量报告（2026-04-21）

20 个新物体的 hybrid_score 全部低于 0.6，远低于 golden config 门槛 0.78：

- 最高：obj_052 = 0.582（needs_fix）
- 中位数：~0.46
- 最低：obj_059 = 0.368（reject）
- 所有物体 route = hybrid_reject（仅 obj_052 为 needs_fix）

根因推测：新物体的 Hunyuan3D mesh 质量本身不够好（texture baking 虽然修复了但贴图精度仍偏低），golden config prior 不完全适配新类型物体。

## 已识别的隐性风险

1. 渲染质量低：20 个新物体 hybrid_score 全部 < 0.6（实测），已全部进入训练集。如果训练指标下降，此为首要排查方向。
2. 训练数据比例失衡：新增 60 弱角度 pairs 占总训练集 20%（60/305），如新物体质量差可能引入噪声。
3. yaw000 源图质量：所有 pair 的 source_image 都是 yaw000，VLM loop 未收敛好的物体会污染全部衍生 pair。
4. epoch 29 可能欠拟合：305 pairs vs 245 pairs，但约束固定 epoch 29。

## R1 进度（完整 11 阶段）

| Phase | 说明 | 状态 | 执行者 |
|-------|------|------|--------|
| 1 | 脚本同步到 68/wwz | 完成 | Codex |
| 2 | 评测分析 + 弱角度识别 | 完成 | Codex |
| 3 | Pinned split + Golden config (58 records) | 完成 | Codex |
| 4 | 新物体概念生成 (obj_051~070) | 完成 | Codex |
| 5a | Stage 1-2 (T2I + SAM3) | 完成 | Codex |
| 5b | Stage 3 (Hunyuan3D) — 初次 3GPU 并行 | 失败（GPU1/2 Paint） | Codex |
| 5b' | Stage 3 repaint — GPU0 重跑 13 个物体 | 完成 20/20 有贴图 | Claude Code |
| 5c | VLM loop bootstrap v2 | 完成 20/20 | 自动 pipeline |
| 5d | Rotation export | 完成 20 物体 80 张渲染 | 自动 pipeline |
| 5e | Trainready 构建（需同步脚本） | 完成 60 pairs | Claude Code |
| 6 | wwz→68 传输（本地中继） | 完成 | Claude Code |
| 7 | Augmented dataset 构建 + 验证 | 完成 305/49/56 passed | Claude Code |
| 8 | 训练（30 epoch, 6×H100） | 完成 04-22 00:52~10:30 | Claude Code |
| 9 | 评测（6×H100 并行推理 + metrics） | 完成 488/488 | Claude Code |
| 10 | 比较 (compare.py) | 完成 verdict: inspect | Claude Code |
| 11 | VIEScore 评测（6 GPU 并行，488 pair） | 完成 2026-04-23 | Claude Code |

## 关键路径（wwz 服务器）

| 路径 | 说明 |
|------|------|
| `pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/` | Stage 1-3 资产（修复后） |
| `pipeline/data/evolution_v2_scaling_r1_feedback_objects_20260421_v2/` | VLM loop bootstrap v2（完成） |
| `pipeline/data/v2_scaling_r1_rotation_export_20260421_v2/` | Rotation export v2（完成） |
| `pipeline/data/v2_scaling_r1_trainready_20260421_v2/` | Trainready v2（60 pairs） |
| `pipeline/data/v2_scaling_r1_logs_20260421/` | 所有日志 |
| `configs/seed_concepts/feedback_expansion_r1_objects.json` | 20 个新物体定义 |
| `pipeline/data/golden_config_library.json` | 58 条渲染 prior |

## 关键路径（68 服务器）

| 路径 | 说明 |
|------|------|
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/pinned_split.json` | 冻结 split |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/new_trainready/` | wwz 传输过来的新物体数据 |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/augmented_dataset/` | 合并后训练数据集（305/49/56） |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/validation_report.json` | 验证报告（passed） |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train_command.sh` | 训练脚本 |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_command.sh` | 评测脚本 |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_command.sh` | 比较脚本 |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/train.log` | 训练日志 |
| `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/` | 评测输出（488 推理图 + metrics + compare_report） |
| `$WORKDIR/DiffSynth-Studio/output/v2_scaling_r1_augmented/` | 训练输出（epoch_0029 checkpoint） |
| `$WORKDIR/DiffSynth-Studio/eval_shard_inference.py` | 6 GPU 分片推理脚本 |
| `$WORKDIR/feedback_loop_scripts/` | 已部署的反馈 loop 脚本 |
| `$WORKDIR/SpatialEdit-Bench-Eval/csv_results/ours_feedback/qwen35vl/ours_feedback_rotate_en_vie_score.csv` | R1 VIEScore 合并 CSV（488 rows） |

## 监控命令

```bash
# 查看 R1 compare verdict
ssh zhanghy56_68 "python3 -m json.tool $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json | head -5"
# 查看 R1 metrics summary
ssh zhanghy56_68 "cat $WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_summary.json"
# 68 GPU 使用
ssh zhanghy56_68 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
```

## 执行分工记录（v2 Scaling R1）

**Codex exec 完成**（Phase 1-5a, 通过 V2_SCALING_TASK.md prompt）：
- Phase 1-4：脚本同步、评测分析、pinned split、golden config、新物体概念生成
- Stage 1-2：T2I + SAM3（20 个物体白底图 + mask）
- Stage 3 初次运行：Hunyuan3D 3GPU 并行（但 GPU1/2 Paint 失败）

**Claude Code 接手**（Phase 5b'-11, Codex exec 因 API 连接问题不可用）：
- Stage 3 诊断：发现 GPU1/2 的 CUDA extension 不兼容导致 Paint 静默失败
- Stage 3 修复：GPU0 单卡重跑 13 个物体
- VLM bootstrap v2 pipeline：编写并启动 `/tmp/v2_repaint_pipeline.sh`
- Trainready 脚本同步：发现 wwz 上的脚本缺少 `--prompts-json` 参数，scp 同步本地版本
- 手动运行 trainready（pipeline 脚本 export 完成后挂起，手动接管）
- wwz→68 传输 + augmented dataset 构建 + 验证 + 训练启动
- VIEScore 6 卡并行评测 + 与 exp5 per-angle 对比
