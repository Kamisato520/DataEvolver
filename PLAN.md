# 外部反馈 Loop v2：评测反馈驱动数据集扩容

## Summary
- 外部反馈 loop 的核心目标改为：把评测失败模式反馈到数据集构建 pipeline，通过增加新物体和弱角度 pairs 做 scaling，而不是修改训练策略。
- 默认服务器分工固定为：`wwz` 执行新物体 Stage 1→VLM loop→rotation export；`68` 接收渲染产物后合并 augmented dataset、训练、评测。
- 已验证现有脚本支持：`export_rotation8_from_best_object_state.py --rotations`、`build_rotation8_trainready_dataset.py --target-rotations`；新增工作重点是 pinned split、反馈诊断、golden config、augmented dataset 合并与状态记录。

## Key Changes
- 新增反馈诊断：从 Test Set / SpatialEdit 指标中同时输出 per-angle 与 per-object 分析，区分“系统性弱角度”和“少数物体拉低角度”，并识别全角度都差的物体。
- 新增 pinned split 机制：一次性从现有 full50 split 生成 `pinned_split.json`，冻结旧 50 个物体的 train/val/test；后续新增物体只进入 train，避免 `seed=42 + shuffle` 因扩容重排旧 split。
- 新增 augmented dataset 构建：保留 baseline val/test 和旧 train pairs，只追加新物体在弱角度上的 train pairs；生成 `augmented_manifest.json`，记录 expected pair counts、追加对象、追加角度和来源路径。
- 新增 golden config prior：从历史高质量渲染 metadata 中提取 warm-start 配置，质量门槛默认 `hybrid_score >= 0.78`；数值字段取中位数，排除 `object.yaw_deg`，`material.*` 按 category 聚合并提供全局中位数 fallback。
- 新增新物体概念生成：读取现有 `prompts.json` / object list 去重，ID 从 `obj_051` 递增；category 分布按现有 50 物体分布采样或匹配，输出给 Stage 1 使用的 objects file。
- 修改 VLM bootstrap 入口：`bootstrap_scene_yaw000_objects.py` 增加 `--render-prior-library`，对新物体按 category 使用 golden config 初始化，减少 VLM loop 迭代次数。
- 修改验证与比较：`validate_dataset.py` 增加 `--pinned-split-path` 与 manifest 驱动 pair count 校验；`compare.py` 增加强角度 regression guard，防止弱角度扩容后原本表现好的角度被稀释退化。

## Workflow
- Round 开始时，读取上轮评测结果，生成 `dataset_feedback_plan.json`：包含弱角度、强角度 guard、建议新增物体数、category 配额、需要追加的 target rotations。
- 在 `wwz` 上执行数据构建：Stage 1 生成新物体概念/T2I/SAM2/Hunyuan3D，使用 golden config warm-start 进入 yaw000 VLM loop，再只导出 `0 + weak target rotations`。
- 用 `rsync/scp` 把 wwz 的新物体 rotation 渲染产物、metadata、manifest 同步到 68 的 `$WORKDIR/feedback_loop_runs/<experiment_id>/round_<N>/`.
- 在 68 上构建 augmented train-ready dataset：旧 full50 split 保持不动，新物体弱角度 pairs 只追加到 train；通过 pinned split 和 manifest validation 后再训练评测。
- 训练 wrapper 仅使用已有训练策略参数化运行，不引入训练策略搜索；评测后更新 `FEEDBACK_STATE.json` 并进入下一轮。

## FEEDBACK_STATE v2
- 顶层记录：`experiment_id`、`round`、`baseline_dataset`、`baseline_checkpoint`、`current_dataset`、`status`、`created_at`、`updated_at`。
- `v2_dataset_construction` 记录：`wwz_run_root`、`server68_run_root`、`pinned_split_path`、`golden_config_path`、`new_object_ids`、`object_count_delta`、`weak_target_rotations`、`pair_count_delta`、`augmented_manifest_path`。
- `diagnostics` 记录：`weak_angles`、`strong_angles`、`all_angle_bad_objects`、`object_angle_outliers`、`per_angle_delta`、`per_object_delta`、`verdict`。
- `cost` 记录：`estimated_hours_min/max`、`actual_stage_durations`、`vlm_loop_rounds_avg`、`vlm_loop_rounds_by_object`，用于判断 golden prior 是否减少迭代。

## Test Plan
- 本地静态检查：`python -m py_compile scripts/feedback_loop/*.py scripts/build_rotation8_trainready_dataset.py`，以及新增 shell wrapper 的 `bash -n`。
- Toy 数据测试：验证 pinned split 不移动旧物体、新物体只进入 train、动态 pair counts 来自 `augmented_manifest.json`、`object.yaw_deg` 不进入 golden prior 聚合。
- 指标解析测试：用已有 exp4/exp5 或 fixture 跑 compare，确认输出 weak angles、per-object outliers、strong-angle regression guard。
- wwz 小样本 dry-run：用 1-2 个新物体、2 个目标角度跑 Stage 1→VLM loop→rotation export，检查 golden prior 被使用、metadata 完整、渲染路径可同步。
- 68 smoke：同步小样本产物后构建 augmented dataset，运行 validate，确认旧 val/test 不变、source/target 指向 `views/`，再生成训练/评测命令但不改训练策略。

## Assumptions
- 默认每轮新增 20 个物体；若 plan agent 估算超过 12 小时或弱角度超过 4 个，则本轮 capped 为 20 个物体和最多 4 个弱角度。
- 20 个新物体端到端成本估计为 5-12 小时：Stage 1 约 2 分钟，T2I 约 10 分钟，SAM2 约 5 分钟，Hunyuan3D 约 50 分钟/20 obj，Stage 4 + VLM loop 约 30 分钟/obj 且 warm-start 后可能减半，rotation export 约 20 分钟。
- 68 只负责训练和评测；除非另行确认 Blender、SAM2、Hunyuan3D、Qwen3.5、Qwen-Image-2512 已完整部署，否则不在 68 上执行数据构建 pipeline。
- 训练策略、LoRA rank、epoch、optimizer、checkpoint 选择保持原 v1 设定不变；反馈 loop 只改变数据规模、对象覆盖和弱角度 pairs 分布。
