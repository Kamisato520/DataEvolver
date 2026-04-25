# 实验

## 1. 评测设置

### 1.1 评测基准

所有实验统一在 **SpatialEdit-Bench** 上评测，共 488 个 rotate pair，由 61 个物体 × 8 个 azimuth 角度（0°/45°/90°/135°/180°/225°/270°/315°）构成。评测物体与训练物体不重叠（object-disjoint）。

### 1.2 评测指标

**传统像素级/感知指标**（camera\_level 评测，基于 Ground Truth 图像对比）：

| 指标 | 方向 | 说明 |
|------|------|------|
| PSNR | ↑ | 峰值信噪比，衡量像素级保真度 |
| SSIM | ↑ | 结构相似性，衡量局部结构保留 |
| LPIPS | ↓ | 感知损失，衡量人眼可见差异 |
| CLIP-I | ↑ | CLIP 图像语义相似度，衡量语义一致性 |
| DINO | ↑ | DINOv2 特征相似度，衡量视觉细节保真度 |
| FID | ↓ | Fréchet 感知距离，衡量分布级图像质量 |

**VLM 语义指标**（object\_level 评测，由 Qwen3.5-VL-8B 无参考评审，488 pair 分 6 shard 并行）：

| 指标 | 方向 | 说明 |
|------|------|------|
| Score\_view | ↑ | 视角正确性评分（旋转是否到达目标角度） |
| Score\_cons | ↑ | 外观一致性评分（与 source 图像的物体外观一致程度） |
| VIE Overall | ↑ | Score\_view 与 Score\_cons 的综合得分 |

---

## 2. 训练设置

| 超参 | 值 |
|------|----|
| 基础模型 | Qwen-Image-Edit-2511 |
| LoRA rank | 32 |
| 学习率 | 1e-4 |
| 训练 epoch | 30（统一取 epoch 29 checkpoint） |
| 训练框架 | DiffSynth-Studio + accelerate |
| GPU 配置 | 6 × H100（68 服务器） |
| Prompt 版本 | v3：`Rotate this {obj} clockwise from front view to {view}.` |

训练脚本：`code/training/train_clockwise.py / .sh`。

---

## 3. 总体结果

以下为 exp5（基线，ours\_objinfo）与 v2 R1（ours\_feedback，+60 弱角度 pairs）在 SpatialEdit-Bench 488 对上的总体指标对比，数据来源：[`eval/v2_scaling_r1/compare_report.json`](../eval/v2_scaling_r1/compare_report.json) 及 [`docs/R1_vs_exp5_spatialedit_bench.md`](../docs/R1_vs_exp5_spatialedit_bench.md)。

| 指标 | exp5 (baseline) | v2 R1 | Δ | 方向 |
|------|:-:|:-:|:-:|------|
| PSNR ↑ | 16.63 | **16.68** | +0.05 | 提升 |
| SSIM ↑ | 0.7296 | **0.7310** | +0.0014 | 提升 |
| LPIPS ↓ | 0.2564 | **0.2546** | −0.0018 | 提升 |
| CLIP-I ↑ | 0.9050 | **0.9499** | **+0.0449** | 大幅提升 |
| DINO ↑ | **0.8895** | 0.8837 | −0.0058 | 退化 |
| FID ↓ | **50.83** | 55.93 | +5.10 | 退化 |
| Score\_view ↑ | 0.7705 | **0.7828** | **+0.0123** | 提升 |
| Score\_cons ↑ | **0.9709** | 0.9676 | −0.0033 | 微降 |
| VIE Overall ↑ | 0.8649 | **0.8703** | +0.0054 | 提升 |

---

## 4. Per-Angle 分析

### 4.1 PSNR 与 DINO per-angle 对比

| 角度 | PSNR_exp5 | PSNR_R1 | ΔPSNR | DINO_exp5 | DINO_R1 | ΔDINO |
|------|:---------:|:-------:|:-----:|:---------:|:-------:|:-----:|
| 0° | 17.09 | 17.12 | +0.02 | 0.9014 | 0.8914 | −0.010 |
| 45° | 16.40 | 16.46 | +0.06 | 0.9086 | 0.8979 | −0.011 |
| 90° | 17.81 | 17.93 | **+0.13** | 0.9143 | 0.9166 | +0.002 |
| 135° | 16.08 | 16.12 | +0.03 | 0.9033 | 0.8866 | −0.017 |
| 180° | 16.98 | 17.19 | **+0.22** | 0.9049 | 0.9026 | −0.002 |
| 225° | 15.73 | 15.78 | +0.06 | 0.8640 | 0.8500 | −0.014 |
| 270° | 16.64 | 16.62 | −0.02 | 0.8380 | 0.8423 | +0.004 |
| 315° | 16.33 | 16.26 | −0.07 | 0.8818 | 0.8821 | +0.000 |

### 4.2 Score\_view per-angle 对比

| 角度 | Score\_view exp5 | Score\_view R1 | Δ |
|------|:----------------:|:--------------:|:-:|
| 0° | 0.9344 | 0.9508 | +0.016 |
| 45° | 0.6066 | 0.5738 | −0.033 |
| 90° | 1.0000 | 1.0000 | 0 |
| 135° | 0.5574 | 0.6066 | **+0.049** |
| 180° | 0.8197 | 0.8689 | **+0.049** |
| 225° | 0.7541 | 0.7541 | 0 |
| 270° | 0.9016 | 0.8852 | −0.016 |
| 315° | 0.5902 | 0.6230 | +0.033 |

**关键发现：**
- PSNR 在 8 个角度中有 6 个提升，提升最显著的是弱角度 180°（+0.22）和 90°（+0.13），与扩容策略一致。
- Score\_view 在 135° 和 180° 大幅提升（各 +0.049），315° 也意外被带动（+0.033），说明弱角度扩容有一定泛化效应。
- 90° Score\_view 在 exp5 已饱和（1.0），无提升空间。270° Score\_view 轻微倒退（−0.016）。
- DINO 退化集中在 0°/45°/135°/180°/225° 等多个角度，与外观一致性指标（Score\_cons）微降一致。

---

## 5. R1 verdict = `inspect` 归因

### 5.1 触发条件

`compare.py` 的 verdict 判断逻辑如下：弱角度 PSNR 均值提升（Δweak\_PSNR = +0.049），但 45° 角度的 DINO 退化达 −0.010，超过 strong-angle regression guard 阈值（−0.008）：

```json
"strong_angle_regressions": [
  {"angle": "45", "metric": "dino", "delta": -0.00994, "threshold": -0.008}
]
```

触发 regression guard → verdict = `inspect`（而非 `continue`）。

### 5.2 根因分析

**直接原因**：20 个新物体的渲染质量低（hybrid\_score 全部 < 0.6，而 golden config 门槛为 0.78），引入噪声数据，导致视觉细节类指标（DINO、FID）退化。

**间接原因**：
1. Hunyuan3D Stage 3 的 texture baking 修复（GPU0 单卡重跑）解决了 shape-only 问题，但贴图精度仍不足，VLM bootstrap 无法将 hybrid\_score 收敛到高质量区间。
2. golden\_config\_library 的 prior（来自前 50 个物体）对新类别（vehicle、street\_object）的光照/材质适配性不足。
3. 新增 60 pairs 占训练集 20%（60/305），低质量数据比例较高，对强角度数据产生干扰。

### 5.3 语义泛化 vs 视觉细节 trade-off

R1 呈现明显的指标分裂：语义泛化类指标（CLIP-I +0.045、Score\_view +0.012、PSNR +0.05）整体提升，而视觉细节类指标（DINO −0.006、FID +5.10、Score\_cons −0.003）轻微退化。这表明**弱角度扩容方向本身是正确的（模型习得了更广泛的语义方向感），但渲染质量是当前的主要瓶颈**。

---

## 6. R2 计划

基于 R1 的经验教训，R2 将引入以下改进：

| 改进项 | 说明 |
|--------|------|
| 渲染质量门控 | Rotation export 前过滤 hybrid\_score < 0.6 的物体，拒绝低质量数据进入训练集 |
| VLM bootstrap 参数调整 | 扩大 max\_rounds（当前 10），调整 plateau 参数，提升新物体收敛率 |
| Stage 3 多 GPU 风险规避 | 强制单 GPU 运行 Hunyuan3D Paint，或事先在每个 GPU 做 smoke test |
| 替换低质量物体 | 对 hybrid\_score < 0.5 的物体（如 obj\_059）重新生成或更换类别 |

目标：在 R2 中实现弱角度提升的同时，DINO 不再触发 strong-angle regression guard，从而获得 verdict = `continue`，进入下一轮迭代。
