# 00_OVERVIEW — 项目故事线骨架

> 本文档为报告 Introduction / Abstract 提供核心素材。报告撰写时直接取用本文件中的段落与数据，结合 [01_method.md](01_method.md)、[03_experiments.md](03_experiments.md) 构成完整叙事。

---

## 终极目标

本研究的终极愿景是实现一键式合成数据集构建：通过一条指令 `/data-build <数据集描述>`，自动完成文本提示扩充、文生图合成（T2I）、3D 重建与渲染、VLM 质检，直接产出训练就绪的旋转视图数据集。**数据合成是核心目的；LoRA 微调仅作为数据质量的端到端验证手段。**

---

## 核心 Motivation

**问题一：旋转视图编辑模型在弱角度上表现不稳定。**
当前主流图像编辑模型（如 Qwen Image Edit）在固定轴旋转任务上，对 azimuth 90°、180°、270° 等"正交"弱角度的生成质量明显弱于 45°/135° 等斜向视角。这类模型缺乏精准角度条件，主要依赖 prompt 语义驱动，在强旋转角度下极易丢失物体外观一致性或生成视角不正确的结果。现有评测体系（SpatialEdit-Bench）已验证，基线模型在 180° 的 Score\_view 仅为 0.82，而 0° 达到 0.93，差距显著。

**问题二：高质量旋转视图数据集的人工构建代价极高。**
现有方法依赖人工拍摄或手工渲染的配对数据，覆盖角度有限，难以快速扩充弱角度样本。本研究提出通过全自动 3D 渲染 pipeline（T2I → 3D 重建 → Blender 渲染 → VLM 质检）合成覆盖 360° 的配对数据，并利用反馈 Loop 识别弱角度、有针对性地增加相应训练 pair，以低成本提升弱角度表现。

---

## 方法概览

本研究采用三层架构，从数据合成、反馈迭代到模型验证形成完整闭环：

**数据 Pipeline（Stage 1–5）：**

```
文本扩充 (Stage 1, Qwen-LM)
  ↓
文生图白底图 (Stage 2, Qwen Image Edit / T2I)
  ↓
SAM3 前景分割 (Stage 2.5)
  ↓
Hunyuan3D 3D 重建 (Stage 3，含 texture baking)
  ↓
Blender 物体旋转渲染 (Stage 4，yaw_deg 7 角度：45°~315°)
  ↓
VLM 质检 Loop (Stage 5, Qwen3.5-VL-8B, hybrid_score 门控)
  ↓
trainready 配对 CSV 输出
```

**反馈 Loop v2（评测驱动数据集扩容）：**

```
SpatialEdit-Bench 评测（传统指标 + VIEScore）
  ↓
compare.py → 识别弱角度（per-angle PSNR 最低 3 个）
  ↓
生成 dataset_feedback_plan.json → 新物体概念生成
  ↓
wwz 服务器：新物体 Stage 1-5 pipeline
  ↓
传输到 68 服务器 → build_augmented_rotation_dataset.py
  ↓
validate_dataset.py 可训练性检验（pinned split 校验）
  ↓
训练（68 服务器，6×H100，epoch 29）
  ↓
评测（488 pair，6 GPU 并行）
  ↓
compare.py → verdict（accept / inspect / reject）
  ↓
下一轮（增加物体或调整策略）
```

详细架构与脚本清单见 [feedback_loop_internals.md](feedback_loop_internals.md)。

---

## R1 关键结果

在 SpatialEdit-Bench 488 个旋转配对（61 个物体 × 8 个角度）上，反馈 Loop v2 Scaling R1 与 exp5 baseline 相比，verdict 为 **`inspect`**。

| 指标 | exp5 (baseline) | v2 R1 | delta |
|------|:-:|:-:|:-:|
| PSNR ↑ | 16.63 | 16.68 | +0.05 |
| SSIM ↑ | 0.7296 | 0.7310 | +0.001 |
| LPIPS ↓ | 0.2564 | 0.2546 | −0.002 |
| CLIP-I ↑ | 0.9050 | 0.9499 | **+0.045** |
| DINO ↑ | **0.8895** | 0.8837 | −0.006 |
| FID ↓ | **50.83** | 55.93 | +5.10 |
| Score\_view ↑ | 0.7705 | 0.7828 | **+0.012** |
| Score\_cons ↑ | **0.9709** | 0.9676 | −0.003 |
| VIE Overall ↑ | 0.8649 | 0.8703 | +0.005 |

**正向信号**：CLIP-I +0.045、Score\_view +0.012，目标弱角度（180° Score\_view +0.049、135° +0.049、315° +0.033）同步提升，语义泛化能力改善。PSNR 在 7/8 角度全部提升。

**退化信号**：DINO 在 4 个角度退化（0°/45°/135°/225°），FID +5.10，Score\_cons −0.003，触发 strong-angle regression guard。根因：R1 引入的 20 个新物体渲染质量低（hybrid\_score 全部 < 0.6），引入噪声污染外观一致性。

**结论**：Scaling 方向正确——增加弱角度新物体能有效提升语义视角正确性；但渲染质量门控是 R2 必须解决的瓶颈。详细 per-angle 对比见 [R1\_vs\_exp5\_spatialedit\_bench.md](R1_vs_exp5_spatialedit_bench.md)。

---

## 报告章节映射

| 报告章节 | 内容 | 对应素材 |
|----------|------|---------|
| Section 1 Introduction | 问题背景、motivation、贡献点 | 本文档 + [01\_method.md](01_method.md)（待补） |
| Section 2 Method | Pipeline 详述、反馈 Loop 机制 | [01\_method.md](01_method.md) + [feedback\_loop\_internals.md](feedback_loop_internals.md) |
| Section 3 Dataset | 数据集构成、新物体生成策略 | [02\_dataset.md](02_dataset.md)（待补）+ `../eval/v2_scaling_r1/trainready/` |
| Section 4 Experiments | 评测设置、Table 1、per-angle 分析 | [03\_experiments.md](03_experiments.md)（待补）+ `../eval/derived/` |
| Section 5 Discussion | 结论、局限性、R2 方向 | [03\_experiments.md](03_experiments.md) 末尾 + 本文档 R1 结论段 |
| Appendix | Checkpoint 路径、服务器配置 | `../CHECKPOINTS.md` + [server\_reference.md](server_reference.md) |
