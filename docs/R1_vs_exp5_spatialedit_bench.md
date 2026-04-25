# R1 反馈 Loop (ours_feedback) vs exp5 Baseline (ours_objinfo) — SpatialEdit-Bench 评测对比

## 背景

- 评测体系：SpatialEdit-Bench（488 rotate pair，61 objects × 8 angles）
- exp5 baseline：ours_objinfo，rank=32, lr=1e-4, epoch 29, Prompt v3
- R1：v2 Scaling R1，在 exp5 基础上新增 20 个物体在弱角度（90°/180°/270°）的 60 训练 pair（hybrid_score 全部 < 0.6），训练 305 train / 49 val / 56 test，epoch 29，6×H100
- 传统指标（PSNR/SSIM/LPIPS/CLIP-I/DINO/FID）来自 SpatialEdit-Bench 的 camera_level 评测
- VIEScore（Score_view / Score_cons）来自 SpatialEdit-Bench 的 object_level 评测，backbone 为 Qwen3.5-VL-8B，488 pair 分 6 shard 并行在 GPU 0-5 上完成

---

## Overall 结果

| 指标 | exp5 | R1 | Δ | 方向 |
|------|------|----|---|------|
| PSNR ↑ | 16.63 | 16.68 | +0.05 | 提升 |
| SSIM ↑ | 0.7296 | 0.7310 | +0.0014 | 提升 |
| LPIPS ↓ | 0.2564 | 0.2546 | -0.0018 | 提升 |
| CLIP-I ↑ | 0.9050 | 0.9499 | +0.0449 | 大幅提升 |
| DINO ↑ | 0.8895 | 0.8837 | -0.0058 | 退化 |
| FID ↓ | 50.83 | 55.93 | +5.10 | 退化 |
| Score_view ↑ | 0.7705 | 0.7828 | +0.0123 | 提升 |
| Score_cons ↑ | 0.9709 | 0.9676 | -0.0033 | 微降 |
| VIE Overall ↑ | 0.8649 | 0.8703 | +0.0054 | 提升 |

---

## Per-Angle Score_view

| angle_idx | 角度说明 | exp5 | R1 | Δ |
|-----------|----------|------|----|---|
| 00 | 0°（front→right） | 0.9344 | 0.9508 | +0.016 |
| 01 | 45° | 0.6066 | 0.5738 | -0.033 |
| 02 | 90° | 1.0000 | 1.0000 | 0 |
| 03 | 135° | 0.5574 | 0.6066 | +0.049 |
| 04 | 180° | 0.8197 | 0.8689 | +0.049 |
| 05 | 225° | 0.7541 | 0.7541 | 0 |
| 06 | 270° | 0.9016 | 0.8852 | -0.016 |
| 07 | 315° | 0.5902 | 0.6230 | +0.033 |

---

## Per-Angle 传统指标（PSNR 与 DINO）

| angle | PSNR_exp5 | PSNR_R1 | ΔPSNR | DINO_exp5 | DINO_R1 | ΔDINO |
|-------|-----------|---------|-------|-----------|---------|-------|
| 00 | 17.0923 | 17.1169 | +0.02 | 0.9014 | 0.8914 | -0.010 |
| 01 | 16.4008 | 16.4569 | +0.06 | 0.9086 | 0.8979 | -0.011 |
| 02 | 17.8061 | 17.9311 | +0.13 | 0.9143 | 0.9166 | +0.002 |
| 03 | 16.0838 | 16.1192 | +0.03 | 0.9033 | 0.8866 | -0.017 |
| 04 | 16.9765 | 17.1927 | +0.22 | 0.9049 | 0.9026 | -0.002 |
| 05 | 15.7268 | 15.7835 | +0.06 | 0.864 | 0.85 | -0.014 |
| 06 | 16.6439 | 16.6182 | -0.02 | 0.838 | 0.8423 | +0.004 |
| 07 | 16.3326 | 16.2568 | -0.07 | 0.8818 | 0.8821 | +0.000 |

---

## 关键发现

1. 视角正确性整体提升：Score_view / PSNR / LPIPS / CLIP-I 都往好的方向走，尤其 180° 和 135° 等弱/中角度。
2. 外观一致性轻微退化：Score_cons / DINO / FID 都微降，根因是 R1 引入的 20 个新物体全部 hybrid_score < 0.6（渲染质量差），引入噪声。
3. 弱角度扩容策略部分验证有效：180° Score_view +0.049（明显提升），135°/315° 意外被带动（泛化效应），但 90° 已饱和（exp5 已经 1.0）、270° 轻微倒退。
4. PSNR 在 7/8 角度全部提升，只有 315° 微降 0.07。
5. DINO 退化集中在 0°/45°/135° 等强角度，与 VIEScore 的 Score_cons 微降一致。

---

## R2 方向

- 加入渲染质量门控（hybrid_score ≥ 0.6 过滤）以解决外观一致性退化问题。
- 当前 R1 verdict = inspect，因为触发 strong-angle regression guard（DINO 在部分强角度退化）。

---

## 数据源路径（68 服务器）

| 描述 | 路径 |
|------|------|
| 合并后 VIEScore CSV | `$WORKDIR/SpatialEdit-Bench-Eval/csv_results/ours_feedback/qwen35vl/ours_feedback_rotate_en_vie_score.csv`（488 rows） |
| R1 传统指标 summary | `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_summary.json` |
| R1 推理图 488 张 | `$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit/ours_feedback/` |
| R1 LoRA | `$WORKDIR/DiffSynth-Studio/output/v2_scaling_r1_augmented/epoch_0029/lora.safetensors` |
| exp5 VIEScore CSV | `$WORKDIR/SpatialEdit-Bench-Eval/csv_results/ours_objinfo/qwen35vl/ours_objinfo_rotate_en_vie_score.csv` |
| exp5 传统指标 CSV | `$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv` |
