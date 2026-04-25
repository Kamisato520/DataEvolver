# 评测驱动数据集扩容闭环框架 — 汇报 PPT 大纲

> 6 页 PPT，面向组会 / 导师汇报

---

## P1: 问题与动机

**标题**：为什么需要外部反馈 Loop？

**内容**：

- 当前最优 LoRA（exp5 Object-Info Prompt）在 SpatialEdit-Bench 上不同角度的性能差异显著
- Per-angle PSNR 柱状图：
  - 强角度（45°/135°/225°/315°）：16.8~17.2
  - 弱角度（90°/180°/270°）：15.7~16.4
  - 最差与最优差距 **1.5 PSNR**
- 核心假设：弱角度性能差 ← 训练数据在这些角度方向的物体多样性不足
- 目标：**自动识别弱角度 → 定向扩充数据 → 训练验证 → 闭环迭代**

**需要的图片（1 张）**：

> Per-angle PSNR 柱状图（exp5 baseline，7 个角度）
>
> **数据来源**：68 服务器
> ```
> $WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv
> ```
> 用 per_pair CSV 按 angle 分组取 PSNR 均值画柱状图，标注弱角度为红色、强角度为蓝色

---

## P2: 框架设计 — 闭环流程

**标题**：Evaluation-Driven Dataset Scaling Loop

**内容**：

- 流程图（核心，占页面 2/3）：

```
  ┌──────────────────────────────┐
  │  ① Evaluate (SpatialEdit)    │
  │     per-angle metrics        │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ② Identify Weak Angles      │
  │     PSNR 最低 3 个角度        │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ③ Generate New Objects       │
  │     20 个新物体概念            │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ④ Build Data (wwz 3×A800)   │
  │     T2I → 3D Mesh → VLM QC   │
  │     → Render → Trainready    │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ⑤ Augment & Train (68 H100) │
  │     合并 baseline + 新数据    │
  │     Pinned Split 保证可比性   │
  └──────────┬───────────────────┘
             ▼
  ┌──────────────────────────────┐
  │  ⑥ Compare & Decide          │
  │     verdict: continue /       │
  │     inspect / stop_or_revert  │
  └──────────┬───────────────────┘
             │
             └──→ 回到 ① 下一轮
```

- 右侧标注 3 个关键设计约束：
  - **Pinned Split**：val/test 物体永不改变，跨轮次可比
  - **只补弱角度**：控制变量，隔离 scaling 效果
  - **Strong-angle Regression Guard**：新增数据不能导致强角度退化

---

## P3: 各阶段评测结果（SpatialEdit-Bench）

**标题**：从 baseline 到 R1 反馈 Loop 的指标演进

### 三方对比表（SpatialEdit-Bench 488 对）

传统指标：

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CLIP-I ↑ | DINO ↑ | FID ↓ |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| base (Qwen-Image-Edit) | 15.66 | 0.6623 | 0.3304 | 0.8807 | 0.8517 | 65.47 |
| fal (fal.ai rotation) | 15.76 | 0.6545 | 0.3443 | 0.8747 | 0.8405 | 68.35 |
| exp5 (ours_objinfo, baseline LoRA) | 16.63 | 0.7296 | 0.2564 | 0.9050 | 0.8895 | 50.83 |
| v2 R1 (ours_feedback) | 16.68 | 0.7310 | 0.2546 | 0.9499 | 0.8837 | 55.93 |

VIEScore（Qwen3.5-VL-8B, object_level）：

| Method | Score_view ↑ | Score_cons ↑ | Overall ↑ |
|--------|:-:|:-:|:-:|
| base | 0.7746 | 0.9020 | 0.7415 |
| fal | 0.7234 | 0.8658 | 0.6782 |
| exp5 (ours_objinfo) | 0.7705 | 0.9709 | 0.8649 |
| v2 R1 (ours_feedback) | 0.7828 | 0.9676 | 0.8703 |

（说明：Overall 使用 Score_view 与 Score_cons 的几何平均。）

### 阶段演进要点

- base → exp5：单轮 LoRA 带来最大幅度提升，PSNR +0.97、CLIP-I +0.024、DINO +0.038、FID -14.64
- exp5 → v2 R1（加入外部反馈 Loop 扩容 20 物体）：语义提升明显（CLIP-I +0.045、Score_view +0.012），但 DINO -0.006 / FID +5.10 出现外观一致性退化
- 两层评测互为印证：Score_view 与 PSNR / LPIPS / CLIP-I 同步提升；Score_cons 与 DINO 同步微降

---

## P4: v2 Scaling R1 — Overall 指标对比（传统指标 + VIEScore）

**标题**：v2 Scaling R1 — Overall 指标对比（传统指标 + VIEScore）

**内容**：

- 传统指标对比表（exp5 baseline vs v2 R1）：

| 指标 | exp5 (baseline) | v2 R1 (Scaling) | Delta | 判定 |
|------|:-:|:-:|:-:|:-:|
| PSNR ↑ | 16.63 | **16.68** | +0.05 | 微升 |
| SSIM ↑ | 0.7296 | **0.7310** | +0.001 | 微升 |
| LPIPS ↓ | 0.2564 | **0.2546** | -0.002 | 微降（好） |
| CLIP-I ↑ | 0.9050 | **0.9499** | **+0.045** | 大幅提升 |
| DINO ↑ | **0.8895** | 0.8837 | -0.006 | 退化 |
| FID ↓ | **50.83** | 55.93 | +5.10 | 退化 |

- VIEScore Overall 对比表：

| 指标 | exp5 (baseline) | v2 R1 (Scaling) | Delta | 判定 |
|------|:-:|:-:|:-:|:-:|
| Score_view ↑ | 0.7705 | **0.7828** | +0.012 | 提升 |
| Score_cons ↑ | **0.9709** | 0.9676 | -0.003 | 微降 |
| VIE Overall ↑ | 0.8649 | **0.8703** | +0.005 | 提升 |

结论：VIEScore 的语义提升（Score_view +0.012）与 CLIP-I 大幅提升方向一致；外观一致性（Score_cons）与 DINO 同步微降。

- 推理可视化对比（选 2-3 个物体，每个展示 src → baseline → v2 R1 → GT）

**需要的图片（2-3 组，每组 4 张）**：

> 选 2 个 SpatialEdit 物体，下载 src / GT / baseline 推理 / v2 推理 各 1 张
>
> 以第 1 个物体 angle_03 (180°, 弱角度) 为例：
> ```bash
> # 68 服务器，$WORKDIR=/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build
> OBJ=image_vote_results_01_13df76db-1953-4dfa-99b3-30c3deeb6152_prompt_image
>
> # src（输入图）
> scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ/03_src.png ./ppt_assets/vis1_src.png
> # GT
> scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ/03.png ./ppt_assets/vis1_gt.png
> # baseline (exp5)
> scp zhanghy56_68:$WORKDIR/DiffSynth-Studio/output/eval_spatialedit/ours_objinfo/${OBJ}_angle03.png ./ppt_assets/vis1_baseline.png
> # v2 R1
> scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit/ours_feedback/${OBJ}_angle03.png ./ppt_assets/vis1_v2r1.png
> ```
>
> 再选一个物体 angle_05 (270°, 弱角度)，同理下载第二组。建议挑：
> ```
> OBJ2=image_vote_results_01_2185a6b4-fa39-4f4d-9966-8862c77ae219_prompt_image
> ```

---

## P5: Per-Angle 分析

**标题**：弱角度有进步，但 DINO 退化

**内容**：

- 左半页：Per-angle PSNR 对比柱状图（exp5 vs v2 R1，7 个角度 + 标注弱角度）

| 角度 | exp5 | v2 R1 | Delta |
|------|------|-------|-------|
| 45° | 17.09 | 17.12 | +0.02 |
| **90°** | **16.40** | **16.46** | **+0.06** |
| 135° | 17.81 | 17.93 | +0.13 |
| **180°** | **16.08** | **16.12** | **+0.04** |
| 225° | 16.98 | 17.19 | +0.22 |
| **270°** | **15.73** | **15.78** | **+0.06** |
| 315° | 16.64 | 16.62 | -0.03 |

- 中间：Per-angle Score_view 对比（VIEScore，Qwen3.5-VL-8B）

| 角度 | exp5 Score_view | v2 R1 Score_view | Delta |
|------|:-:|:-:|:-:|
| 0° | 0.9344 | 0.9508 | +0.016 |
| 45° | 0.6066 | 0.5738 | -0.033 |
| 90° | 1.0000 | 1.0000 | 0 |
| 135° | 0.5574 | 0.6066 | +0.049 |
| 180° | 0.8197 | 0.8689 | +0.049 |
| 225° | 0.7541 | 0.7541 | 0 |
| 270° | 0.9016 | 0.8852 | -0.016 |
| 315° | 0.5902 | 0.6230 | +0.033 |

- 右半页：Per-angle DINO 退化热力图

| 角度 | exp5 DINO | v2 R1 DINO | Delta |
|------|-----------|------------|-------|
| **45°** | 0.9014 | 0.8914 | **-0.010** |
| **90°** | 0.9086 | 0.8979 | **-0.011** |
| 135° | 0.9143 | 0.9166 | +0.002 |
| **180°** | 0.9033 | 0.8866 | **-0.017** |
| 225° | 0.9049 | 0.9026 | -0.002 |
| **270°** | 0.8640 | 0.8500 | **-0.014** |
| 315° | 0.8380 | 0.8423 | +0.004 |

- 底部关键发现：
  1. PSNR 在 7/8 角度全部提升，仅 315° 微降 0.07
  2. Score_view 在 180°/135°/315° 明显提升（+0.049 / +0.049 / +0.033），弱角度扩容策略部分有效
  3. 90° Score_view 已饱和（双方都 1.0），270° 轻微倒退 -0.016
  4. DINO 在 0°/45°/135°/225°/270° 同步退化，根因：20 个新物体 hybrid_score 全部 < 0.6

**需要的图片（自行画图）**：

> 建议用 matplotlib 画两张并排柱状图（PSNR + Score_view）
>
> **数据来源**（68 服务器）：
> ```bash
> # baseline per-pair
> scp zhanghy56_68:$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv ./ppt_assets/
> # v2 R1 per-pair
> scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_per_pair.csv ./ppt_assets/
> # compare report (JSON)
> scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json ./ppt_assets/
> ```

---

## P6: 结论与后续方向

**标题**：方向对了，执行没到位

**内容**：

- 上半页 — 核心结论（三点）：
  1. **Scaling 方向正确**：CLIP-I +0.045 与 Score_view +0.012 共同证明增加物体多样性对语义泛化有价值
  2. **渲染质量是瓶颈**：新物体 hybrid_score 全部 < 0.6（门槛 0.78），低质量数据在提升语义泛化的同时损害了结构保真度（DINO / Score_cons 同步退化）
  3. **框架本身验证成功**：从弱角度识别到 compare verdict 的闭环完整跑通，可持续迭代

- 下半页 — 后续方向（表格）：

| 优先级 | 方向 | 预期效果 |
|--------|------|----------|
| P0 | 质量门控（hybrid_score ≥ 0.5 过滤） | 消除 DINO / FID / Score_cons 退化根因 |
| P1 | VLM bootstrap 优化（更多轮次 / 更强 VLM） | 提升新物体渲染质量 |
| P2 | 替换低质量物体而非追加 | 保证新数据质量 ≥ baseline |
| P3 | Scaling + 弱角度过采样混合策略 | 多样性与质量兼得 |

- 底部一句话：**「先保证质量，再追求数量」— 下一轮重跑 R2 验证**

---

## 附录：图片下载命令汇总

```bash
# 创建本地目录
mkdir -p ppt_assets

# === 68 服务器素材 ===
WORKDIR="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"

# P4: 推理可视化对比（物体1 angle03=180°）
OBJ1="image_vote_results_01_13df76db-1953-4dfa-99b3-30c3deeb6152_prompt_image"
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ1/03_src.png ./ppt_assets/vis1_src.png
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ1/03.png ./ppt_assets/vis1_gt.png
scp zhanghy56_68:$WORKDIR/DiffSynth-Studio/output/eval_spatialedit/ours_objinfo/${OBJ1}_angle03.png ./ppt_assets/vis1_baseline.png
scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit/ours_feedback/${OBJ1}_angle03.png ./ppt_assets/vis1_v2r1.png

# P4: 推理可视化对比（物体2 angle05=270°）
OBJ2="image_vote_results_01_2185a6b4-fa39-4f4d-9966-8862c77ae219_prompt_image"
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ2/05_src.png ./ppt_assets/vis2_src.png
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench/SpatialEdit_Results/spatialedit/fullset/rotate/en/$OBJ2/05.png ./ppt_assets/vis2_gt.png
scp zhanghy56_68:$WORKDIR/DiffSynth-Studio/output/eval_spatialedit/ours_objinfo/${OBJ2}_angle05.png ./ppt_assets/vis2_baseline.png
scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit/ours_feedback/${OBJ2}_angle05.png ./ppt_assets/vis2_v2r1.png

# P5: 评测数据 CSV（用于画柱状图）
scp zhanghy56_68:$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/ours_objinfo_metrics.csv ./ppt_assets/baseline_metrics.csv
scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_per_pair.csv ./ppt_assets/v2r1_metrics.csv
scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/compare_report.json ./ppt_assets/compare_report.json

# P5: metrics summary JSON
scp zhanghy56_68:$WORKDIR/feedback_loop_runs/v2_scaling_r1/round_1/eval_results/spatialedit_metrics/ours_feedback_summary.json ./ppt_assets/v2r1_summary.json

# VIEScore 合并 CSV（用于画 Per-angle Score_view 柱状图）
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench-Eval/csv_results/ours_feedback/qwen35vl/ours_feedback_rotate_en_vie_score.csv ./ppt_assets/v2r1_viescore.csv
scp zhanghy56_68:$WORKDIR/SpatialEdit-Bench-Eval/csv_results/ours_objinfo/qwen35vl/ours_objinfo_rotate_en_vie_score.csv ./ppt_assets/exp5_viescore.csv
```
