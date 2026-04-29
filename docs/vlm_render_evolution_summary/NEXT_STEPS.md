# 下一步研究方向：从 Controller 收敛到资产质量改善

> **前提**：v6b full validation run 已确认 controller 层收敛（见 SUMMARY.md 第 6 节）。本文档描述下一阶段的研究方向和实验计划。
> **注意**："上游资产质量是主瓶颈"目前是最强假设，不是证明结论。下面的方向 1 的目的正是验证或证伪该假设。

---

## Controller 已经解决了什么

经过 v2→v4→v4b→v5→v5b→v6→v6b 七个版本的迭代（v5b 为 smoke-only 验证），以下问题已被 controller 层解决（详见 SUMMARY.md 第 4 节"已证明结论"）：

### ✅ 已解决

1. **零退化保证**：v6 bounded search（每次从 baseline_state 出发）确保 final ≥ confirmed，controller 不会让对象变差。

2. **稳定测量**：p0+p1 双 probe 确认机制，排除了 VLM 单次读数噪声（方差 ~0.05-0.12）对决策的干扰。

3. **局部可解问题已修复**：underexposed_global 类型（obj_002）通过 L_WORLD_EV_UP 成功改善 +0.027。说明当渲染问题是"全局曝光不足"这类简单可解问题时，controller 能正确修复。

4. **动作空间负向 action 已识别**：L_RIM_UP 对 flat_no_rim mesh 系统性有害，已通过 blacklist + v6b Fix A 解决。

5. **9 种 exit reason 覆盖全场景**：controller 对所有对象类型（优质/中等/低分/不稳定/几何损坏）都有对应的处理路径。

---

## Controller 没有解决什么

### ❌ 未解决（不是 controller 的问题）

以下 9/10 对象无法进入 ≥0.77 区间，当前最强解释是**上游资产质量是根本瓶颈**，但这是假设而非已证明事实：

| 类型 | 对象数 | 表现 | 当前假设（未经直接验证） |
|------|--------|------|---------|
| 诊断全 good 低分 | **4** 个（obj_001/003/005/010） | 0.65-0.74，无可改善的诊断信号 | mesh 几何/UV/材质质量不足 |
| flat_no_rim 灯光不可解 | 4 个（obj_004/006/007/008） | lighting_diagnosis=flat_no_rim，所有灯光动作均无效 | mesh 表面平坦，灯光参数不足以产生 rim 高光 |

---

## 主瓶颈的分析

### 为什么 9/10 对象停在 0.65-0.74？

**直接证据（实验观测）**：
- obj_001/003/005/010 的 4 维诊断全部 = "good"（VLM 认为渲染没有显著问题）
- 但分数仍然 <0.77，没有可用的诊断信号来选择 action

**推断（尚未直接验证）**：
- 这些对象使用的是 Stage 2 T2I 生成的参考图 vs Stage 3 Blender 渲染结果
- 两者的差距"超过了灯光参数可以弥合的范围"这一说法目前是推断，不是测量结论

**可能原因（按当前可能性排序，均未实验验证）**：
1. **mesh 几何粗糙**：Stage 3 image-to-3D 生成的 mesh 细节不足，与 T2I 参考图几何差距较大
2. **材质/纹理质量差距**：T2I 纹理细节丰富，3D mesh 贴图/材质简单
3. **当前诊断维度不完整**：可能存在未被 4 维诊断捕获的可修复问题（背景、材质高光等）
4. **评分标准不匹配**：VLM 对"T2I 风格 vs 渲染风格"存在系统性偏差

### 为什么 flat_no_rim 无法用灯光解决？

**直接证据（实验观测）**：
- obj_004/006/007/008 的 lighting_diagnosis = flat_no_rim（4/4 对象）
- v6 使用 L_RIM_UP → 3/4 负收益，1/4 小正收益但低于 eps=0.01
- v6b 改用 L_FILL_DOWN → 同样无法超过 eps

**推断（尚未直接验证）**：
- flat_no_rim 是 mesh 表面曲率不足导致的，而非光源位置/强度问题
- 这是当前最合理的解释，但未通过 mesh 几何测量直接验证

---

## 下一阶段研究方向

### 方向 1：Mesh 质量分析与前置过滤（推荐优先 — 用于验证 H1/H2）

**目标**：通过实际测量 mesh 几何特征，验证或证伪"mesh 质量是主瓶颈"假设。这是目前最重要的实验，因为它将把 H1/H2 假设转化为已证明结论或排除。

**实验方案**：
```python
# mesh 质量指标
- 顶点数、面数密度（低密度 → 细节不足）
- 表面曲率分布（低曲率 → flat_no_rim 高概率）
- 法向量方差（低方差 → 过于平滑）
- Hausdorff 距离（3D mesh vs SDF 重建的对比）
```

**验收标准**：
- 识别出的"低质量 mesh"与 VLM 给出 flat_no_rim 或低分(<0.68)的对象有 ≥70% 重叠
- 形成一个简单的质量得分，阈值过滤

---

### 方向 2：Mesh 后处理与增强

**目标**：对 Stage 3 生成的 mesh 做后处理，提升几何质量。

**可尝试的方法**：

| 方法 | 目标 | 工具 |
|------|------|------|
| 网格细分（Subdivision）| 增加面数，提升曲率细节 | Blender Python API / MeshLab |
| 法向量平滑 | 减少硬边法向量 | Blender smooth shading |
| 表面重建 | 从 mesh 重建更光滑的 SDF | Open3D, PyMCubes |
| 贴图超分辨率 | 提升纹理细节 | Real-ESRGAN, CodeFormer |

**验收标准**：
- flat_no_rim 对象的 lighting_diagnosis 从 flat_no_rim 改变为 good 或其他
- 这 4 个对象的 VLM 分数提升 ≥0.03

---

### 方向 3：Stage 2→3 对齐优化

**目标**：减少 T2I 参考图和 Blender 渲染之间的系统性差距。

**问题诊断**：当前 4 维诊断已经能检测到 color_consistency（obj_002 minor_shift，obj_007 minor_shift），但结构不一致性可能更深。

**实验方案**：
1. 对 10 个对象手动分析 T2I 参考图 vs Stage 3 mesh 的几何差距
2. 量化"T2I 视角"和"Blender 默认渲染视角"的差异
3. 如果视角存在系统性偏差 → 添加 camera angle 优化 action

---

### 方向 4：VLM 评分标定（长期）

**目标**：理解 VLM 评分方差的来源，建立更稳定的评估指标。

**当前问题**：
- VLM 单次评分方差 ~0.05-0.12（p0-p1 span 最大 0.012 in stable case，但跨 round 可达 0.08+）
- 这个方差水平使得 <0.01 的 action delta 完全不可信
- 需要理解：方差来自 VLM 本身的随机性？还是渲染结果的一致性？

**实验方案**：
1. 对同一渲染结果（固定 seed）重复 20 次 VLM 评分，测量方差分布
2. 对不同渲染结果（相同 mesh，不同光照）测量 VLM 评分的可辨度（discriminability）
3. 如果 VLM 本身方差 >> action delta → 需要更多 probes 或不同评分方法

---

### 方向 5：评分函数替换（长期）

**目标**：用传统图像指标（SSIM/LPIPS）替换或补充 VLM 评分，作为更稳定的优化信号。

**当前问题**：VLM 评分捕捉"语义相似性"，但不一定对"渲染参数变化"敏感。

**实验方案**：
1. 计算所有 v6b 渲染结果的 LPIPS（vs T2I 参考图）
2. 对比 LPIPS 和 VLM hybrid_score 的排名相关性
3. 如果 LPIPS 排名更稳定 → 将其引入 hybrid_score 计算（调整权重）

---

## 推荐实验优先级

```
优先级 1（立即可做）:
  [方向1] mesh 质量指标计算 + 与 flat_no_rim 对象的关联分析
  → 可以写一个简单脚本分析 10 个 mesh 的几何特征

优先级 2（1 周内）:
  [方向2] Blender mesh 细分（Subdivision Surface + smooth shading）
  → 对 flat_no_rim 4 对象重新渲染 + VLM 评分，看是否改善

优先级 3（中期）:
  [方向4] VLM 评分方差分析（20 次重复测量）
  → 为后续是否需要增加 probe 数提供数据依据

优先级 4（长期）:
  [方向5] LPIPS 引入 hybrid_score
  [方向3] Stage 2→3 视角对齐分析
```

---

## 快速启动命令参考

### Mesh 质量分析（方向 1）
```bash
# 在服务器上启动分析
ssh wwz
tmux new -s claudecode-research-mesh-quality
cd /aaaidata/zhangqisong/data_build

/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c "
import trimesh, os, numpy as np
mesh_dir = 'pipeline/data/meshes'
for obj_id in sorted(os.listdir(mesh_dir)):
    meshf = f'{mesh_dir}/{obj_id}/{obj_id}.obj'
    if not os.path.exists(meshf): continue
    m = trimesh.load(meshf)
    print(f'{obj_id}: verts={len(m.vertices)}, faces={len(m.faces)}, curvature_std=...')
"
```

### Mesh 细分实验（方向 2）
```python
# Blender Python API: 对 mesh 增加 subdivision surface
import bpy
obj = bpy.context.active_object
mod = obj.modifiers.new(name='SubSurf', type='SUBSURF')
mod.levels = 2
mod.render_levels = 2
bpy.ops.object.modifier_apply(modifier='SubSurf')
```

---

## 结语

v6b 正式标志着 **controller 层完成使命**。当前系统能够：
- 安全地不损害任何对象（零退化，已证明）
- 修复可修复的曝光问题（underexposed_global，已证明）
- 正确识别当前动作空间无法解决的问题（flat_no_rim，已证明）
- 干净地报告每个对象的退出原因和诊断信息

下一步的优先任务是**验证"上游资产质量是主瓶颈"这一假设** — 通过实际测量 mesh 几何特征和诊断与分数的相关性来确认，而不是直接投入大规模 mesh 改善工程。
