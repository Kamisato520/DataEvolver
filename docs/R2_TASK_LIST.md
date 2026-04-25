# Feedback Loop v2 — R2 任务清单

## R1 总结

**实验名称**：v2 Scaling R1（评测驱动数据集扩容首轮）

**做了什么**：从 SpatialEdit-Bench per-angle 评测中识别 3 个弱角度（270°/180°/90°），生成 20 个新物体（obj_051~070），只在弱角度上构建训练 pair，与 baseline 245 pairs 合并为 305 pairs 训练。

**核心结果**：

| 指标 | exp5 baseline | v2 R1 | delta | 判定 |
|------|:-:|:-:|:-:|:-:|
| PSNR ↑ | 16.63 | 16.68 | +0.05 | 微升 |
| SSIM ↑ | 0.7296 | 0.7310 | +0.001 | 微升 |
| LPIPS ↓ | 0.2564 | 0.2546 | -0.002 | 微降（好） |
| CLIP-I ↑ | 0.9050 | **0.9499** | **+0.045** | 大幅提升 |
| DINO ↑ | **0.8895** | 0.8837 | **-0.006** | 退化 |
| FID ↓ | **50.83** | 55.93 | +5.10 | 退化 |

**Verdict**: `inspect` — DINO 在 4 个角度退化触发 strong-angle regression guard。

**R1 教训**：
1. **渲染质量是瓶颈**：20 个新物体 hybrid_score 全部 < 0.6（最高 0.582），相当于往训练集掺了 20% 噪声
2. **Scaling 方向正确**：CLIP-I +0.045 证明物体多样性提升语义泛化
3. **Stage 3 多 GPU 有坑**：GPU1/2 的 Paint 静默失败，必须单 GPU 或逐 GPU smoke test
4. **Golden config prior 阈值被迫下调**：设计门槛 0.78 → 实际只筛到 0.58 的 58 条
5. **VLM bootstrap 收敛太快**：`plateau_window=2` 导致过早停止，质量未充分优化

---

## R2 任务清单

核心改进：**质量门控 + VLM 优化 + 扩大物体预算**。

### Phase 0: R1 复盘 & R2 参数确定

**T0.1 — 分析 R1 中 20 个物体的逐物体指标**
- 目的：找到哪些物体在训练中是"正向贡献"、哪些是"噪声源"
- 方法：从 68 服务器 `ours_feedback_per_pair.csv` 中按 object 聚合 DINO/PSNR，与 baseline 逐物体对比
- 路径：`$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_per_pair.csv`
- 输出：`r1_per_object_analysis.json`（每个新物体的 DINO/PSNR 贡献 delta）

**T0.2 — 确定 R2 质量门控阈值**
- 根据 T0.1 结果，确定 hybrid_score 门槛（候选：0.50 / 0.55 / 0.60）
- 对齐标准：baseline 50 个物体中 hybrid_score 的分布（中位数、P25）
- 输出：写入 `PLAN.md` 的 R2 参数表

**T0.3 — 确定 R2 物体预算**
- R1 用了 20 个物体，全部 < 0.6 → 如果门控 0.55 可能只剩 5-10 个
- R2 预算建议：**40 个新物体**（obj_071~110），预期门控后保留 20+
- 输出：`configs/seed_concepts/feedback_expansion_r2_objects.json`

---

### Phase 1: 质量门控机制开发（本地）

**T1.1 — 新增 `scripts/feedback_loop/quality_gate.py`**
- 功能：读取 VLM bootstrap 输出目录，按 hybrid_score 过滤物体
- 参数：`--min-hybrid-score 0.55 --input-dir <bootstrap_dir> --output-dir <gated_dir>`
- 逻辑：
  - 遍历每个物体的最终 `agent_round*.json`，提取 hybrid_score
  - hybrid_score >= 阈值 → 复制到 gated 目录
  - hybrid_score < 阈值 → 记录到 `rejected_objects.json`（含分数和原因）
  - 输出 `gate_report.json`（通过/拒绝数量、分数分布、通过率）
- 路径：本地 `scripts/feedback_loop/quality_gate.py`

**T1.2 — 修改 rotation export 脚本集成门控**
- 在 `scripts/export_rotation8_from_best_object_state.py` 中增加 `--min-hybrid-score` 参数
- 低于阈值的物体跳过 export，日志输出警告
- 或者：rotation export 前先跑 `quality_gate.py`，只把通过的物体传给 export

**T1.3 — 修改 `build_augmented_rotation_dataset.py` 增加质量元数据**
- 在 augmented_manifest.json 中记录：每个新物体的 hybrid_score、是否通过门控
- 训练集只包含通过门控的物体 pairs

---

### Phase 2: VLM Bootstrap 优化（本地）

**T2.1 — 调整 `bootstrap_scene_yaw000_objects.py` 默认参数**
- 当前：`--max-rounds 10 --plateau-window 2 --plateau-eps 0.01`
- R2 目标：`--max-rounds 15 --plateau-window 3 --plateau-eps 0.008`
- 原因：R1 的 VLM loop 过早收敛（plateau_window=2 意味着连续 2 轮无改善就停），增大窗口和轮次上限让优化更充分

**T2.2 — 改进 golden config prior**
- R1 问题：设计门槛 0.78 实际筛不出，被迫下调到 0.58
- R2 方案：
  - 用 R1 中通过门控的物体的配置更新 golden config library
  - 或者用 baseline 50 物体中 top-20 的配置（hybrid_score 最高的 20 条）
  - 路径：`pipeline/data/golden_config_library.json`（wwz 服务器）

---

### Phase 3: Stage 3 可靠性修复（本地）

**T3.1 — 修改 `pipeline/stage3_image_to_3d.py`：Paint 失败改为显式异常**
- 当前：`try/except` 把 Paint 失败静默 fallback 到 shape-only
- 修改：Paint 失败时抛 `RuntimeError`，由外层决定是否 fallback
- 增加日志：Paint 失败时输出 GPU ID、物体 ID、异常堆栈

**T3.2 — 确定 R2 Stage 3 执行策略**
- R1 教训：GPU1/2 的 CUDA extension 不兼容
- R2 方案：**单 GPU 串行**（GPU 0 only），或先 smoke test 每个 GPU 再并行
- 预计时间：40 物体 × ~2.5min/物体 ≈ 100min（单 GPU）

---

### Phase 4: 新物体概念生成 & 同步（本地 + wwz）

**T4.1 — 生成 R2 新物体概念（40 个）**
- 脚本：`scripts/feedback_loop/build_feedback_expansion_objects.py`
- 参数：`--start-id 71 --count 40 --existing-objects <full50 + R1 的 20>`
- 确保去重：不重复 baseline 50 + R1 的 20 个物体类型
- 输出：`configs/seed_concepts/feedback_expansion_r2_objects.json`

**T4.2 — 同步脚本和概念到 wwz**
- scp 更新后的脚本到 wwz：
  - `scripts/feedback_loop/quality_gate.py`（新增）
  - `scripts/bootstrap_scene_yaw000_objects.py`（参数调整）
  - `pipeline/stage3_image_to_3d.py`（Paint 异常修复）
  - `scripts/export_rotation8_from_best_object_state.py`（门控集成）
  - `configs/seed_concepts/feedback_expansion_r2_objects.json`
- wwz 目标目录：`$WWZ_CODE/`（`/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`）

---

### Phase 5: 数据构建（wwz 服务器）

**T5.1 — Stage 1-2: T2I + SAM3（40 个新物体）**
- 白底图生成 + RGBA mask
- 预计时间：~30min
- 输出目录：`pipeline/data/dataset_v2_scaling_r2_feedback_stage1_assets_<date>/`

**T5.2 — Stage 3: Hunyuan3D（GPU 0 单卡）**
- **关键改进**：单 GPU 串行，避免 R1 的多 GPU Paint 失败
- 先跑 1 个物体 smoke test 验证 Paint 正常
- 预计时间：~100min（40 物体）
- 验证：检查每个 GLB 文件大小（有贴图 > 5MB，无贴图 < 1MB）

**T5.3 — VLM Bootstrap（3×A800 并行，优化参数）**
- 参数：`--max-rounds 15 --plateau-window 3 --plateau-eps 0.008`
- 使用更新后的 golden config prior
- 预计时间：~3-5h（40 物体 / 3 GPU）
- 输出：`pipeline/data/evolution_v2_scaling_r2_<date>/`

**T5.4 — 质量门控**
- 运行 `quality_gate.py --min-hybrid-score 0.55`
- 输出 `gate_report.json`：预期通过 20-30 个物体（40 个中）
- 如果通过率 < 50%（即 < 20 个），考虑降低阈值到 0.50 或追加物体

**T5.5 — Rotation Export（仅通过门控的物体）**
- 只导出通过门控的物体的 yaw000 + 弱角度渲染
- 弱角度沿用 R1：90° / 180° / 270°（除非 R1 per-angle 分析显示需要调整）
- 输出：`pipeline/data/v2_scaling_r2_rotation_export_<date>/`

**T5.6 — Trainready 构建**
- 生成 JSONL pairs
- 输出：`pipeline/data/v2_scaling_r2_trainready_<date>/`

---

### Phase 6: 传输 & 合并（wwz → 本地 → 68）

**T6.1 — wwz→68 传输**
```bash
ssh wwz "tar czf - -C <wwz_trainready_dir> ." | \
  ssh zhanghy56_68 "mkdir -p $WORKDIR/feedback_loop_runs/v2_scaling_r2/round_1/new_trainready && tar xzf - -C $WORKDIR/feedback_loop_runs/v2_scaling_r2/round_1/new_trainready"
```

**T6.2 — Augmented Dataset 构建**
- 脚本：`build_augmented_rotation_dataset.py`
- baseline 245 train + R2 新物体弱角度 pairs = ~305-335 train（取决于门控通过数）
- val/test 不变（pinned split）

**T6.3 — 数据集验证**
- `validate_dataset.py --pinned-split-path pinned_split.json`
- 确保 JSONL 完整、图像存在、split 无泄漏

---

### Phase 7: 训练 & 评测（68 服务器）

**T7.1 — 训练**
- 6×H100，30 epoch，LoRA rank=32，lr=1e-4，取 epoch 29
- 输出：`$WORKDIR/DiffSynth-Studio/output/v2_scaling_r2_augmented/epoch_0029/`

**T7.2 — 6-GPU 并行评测**
- 复用 `eval_shard_inference.py`，6 GPU 分片推理
- **修复 R1 的 eval_meta.json 竞态**：每个 shard 写自己的 `eval_meta_shard{i}.json`，最后合并
- 输出：`$WORKDIR/feedback_loop_runs/v2_scaling_r2/round_1/eval_results/`

**T7.3 — Metrics + Compare**
- 运行 metrics 计算 + `compare.py`（v2 R2 vs exp5 baseline）
- 同时对比 R2 vs R1（三方比较）
- 预期 verdict：如果质量门控有效 → `continue`（DINO 不再退化）

---

### Phase 8: 结果分析 & 文档更新

**T8.1 — 更新 CLAUDE.md**：R2 进度和结果
**T8.2 — 更新 week5 或新建 week6 周报**
**T8.3 — 更新 PLAN.md**：R2 参数、门控阈值、结果

---

## 关键路径总览

| 阶段 | 服务器 | 预计时间 | 依赖 |
|------|--------|---------|------|
| Phase 0: R1 复盘 | 68 + 本地 | 30min | — |
| Phase 1: 质量门控开发 | 本地 | 1h | — |
| Phase 2: VLM 优化 | 本地 | 30min | — |
| Phase 3: Stage 3 修复 | 本地 | 30min | — |
| Phase 4: 概念生成 + 同步 | 本地 → wwz | 30min | Phase 1-3 |
| Phase 5: 数据构建 | wwz | 5-7h | Phase 4 |
| Phase 6: 传输 + 合并 | wwz → 68 | 30min | Phase 5 |
| Phase 7: 训练 + 评测 | 68 | ~12h | Phase 6 |
| Phase 8: 文档 | 本地 | 30min | Phase 7 |

**总预计时间：~20-22h**（大部分是 Phase 5 数据构建 + Phase 7 训练的等待时间）

---

## R2 vs R1 核心差异

| 维度 | R1 | R2 |
|------|----|----|
| 新物体数量 | 20 (obj_051~070) | 40 (obj_071~110) |
| 质量门控 | 无（全部接受） | hybrid_score >= 0.55 |
| VLM max_rounds | 10 | 15 |
| VLM plateau_window | 2 | 3 |
| Stage 3 策略 | 3 GPU 并行（Paint 失败） | GPU 0 单卡串行 |
| Paint 失败处理 | 静默 fallback | 显式异常 |
| Golden config 门槛 | 0.58（被迫下调） | 待定（基于 R1 分析） |
| 预期训练集大小 | 305 (245+60) | ~305-335 (245 + 门控后 pairs) |
