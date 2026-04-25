# ⚠️ 紧急提示 — Codex 主 Agent（v2 Scaling R1 — 需重新安排）

**时间**：2026-04-21 17:05+（用户二次确认）

## ‼️ 用户反馈（核心问题）

> **从 obj_055 开始，后面所有物体的 3D 重建结果都是白色（缺失颜色）。让 codex 重新考虑安排。**

这是 **Stage 3 Hunyuan3D** 的问题，**不是** Stage 4 渲染的光照问题。VLM loop 的 flat_lighting / overexposed tags 只是症状（白色 mesh 在任何光照下都会看起来过曝 / 平光）。

## 证据 — Mesh 文件大小模式

`dataset_v2_scaling_r1_feedback_stage1_assets_20260421/meshes/` 下 20 个 GLB 文件大小：

| 物体 | Size (MB) | 推测 |
|------|-----------|------|
| obj_051 | 1.2 | **无贴图（几何体）** |
| obj_052 | 32.5 | 有贴图 |
| obj_053 | 14.2 | 有贴图 |
| obj_054 | 1.3 | **无贴图** |
| obj_055 | 12.4 | 有几何 + ?（用户说白色） |
| obj_056 | 22.9 | 有贴图 |
| obj_057 | 1.4 | **无贴图** |
| obj_058 | 22.5 | 有贴图 |
| obj_059 | 20.0 | 有贴图 |
| obj_060 | 1.5 | **无贴图** |
| obj_061 | 10.7 | ? |
| obj_062 | 3.5 | 小 |
| obj_063 | 1.3 | **无贴图** |
| obj_064 | 11.4 | ? |
| obj_065 | 15.4 | ? |
| obj_066 | 1.4 | **无贴图** |
| obj_067 | 6.6 | ? |
| obj_068 | 6.8 | ? |
| obj_069 | 1.3 | **无贴图** |
| obj_070 | 12.8 | ? |

**模式 1（geometry-only）**：obj_051, 054, 057, 060, 063, 066, 069 — 每 3 个一个，全是 1.2-1.5 MB。疑似某个 GPU worker 的 Hunyuan3D texture baking 一直失败，只保存了 geometry。

**模式 2（用户报告）**：即使看起来"有贴图"的 obj_055+ 也渲染成白色 — 可能是贴图被写入但**颜色数据全为 255（纯白）** 或 albedo map 指向错误的默认贴图。

## 需要 codex 做的事 ⚡

### 1. 立即（不要继续 Phase 6 传输）
```bash
# 暂停 rotation export 之后的流程
ssh wwz "tmux capture-pane -pt v2_scaling_r1_build -S -300 2>&1 | tail -100"
# 确认 rotation export 已完成后不要自动进 Phase 6
```

### 2. 诊断根因（Stage 3 Hunyuan3D）
```bash
# 查 Stage 3 的 log 看 texture baking 是否报错
ssh wwz "find /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421 -name 'stage3*.log' -o -name 'hunyuan*.log' 2>/dev/null | head -5"

# 查 Stage 3 manifest 看每个物体的 status
ssh wwz "find /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421 -name '*manifest*.json' -o -name '*status*.json' 2>/dev/null"

# 用 trimesh 检查一个"大" mesh 的 material
ssh wwz "/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c \"
import trimesh
m = trimesh.load('/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/meshes/obj_055.glb')
print('scene?', isinstance(m, trimesh.Scene))
if isinstance(m, trimesh.Scene):
    for name, g in m.geometry.items():
        mat = getattr(g.visual, 'material', None)
        print(name, 'material=', type(mat).__name__, 'has_baseColorTexture=', getattr(mat, 'baseColorTexture', None) is not None if mat else 'no_material')
\""
```

### 3. 重新安排（三条路径选一）

**路径 A（快，推荐）：丢弃 obj_055-070，保留 obj_051-054 + 重新生成 16 个新物体**
- 只保留 4 个质量合格的物体（obj_052, 053, 055 如果实际 OK, 056, 058, 059 等，以实际图像判断）
- 用 Hunyuan3D 新 seed 或修复 texture baking pipeline 后重跑 16 个
- 代价：再多 ~4-6 小时

**路径 B（慢，更稳）：整批 20 个物体重跑 Stage 3**
- 先诊断 texture baking 为什么在 GPU 0（假设轮询分配）上失败
- 修复后 20 个全部重新生成
- 代价：~6-8 小时

**路径 C（最保守）：放弃 v2 R1 增量物体，直接用现有 baseline 做 R2**
- 改用其他 scaling 策略（例如同物体更多角度、数据增强等）
- 代价：0（但失去 scaling 信号）

## ‼️ 关键约束

- **不要**把这批有问题的 20 个物体传到 68 服务器训练
- **不要**覆盖 `evolution_v2_scaling_r1_feedback_objects_20260421/` （保留 VLM loop 产物做诊断）
- **保留** tmux `v2_scaling_r1_build` 的最终日志（`tmux capture-pane -pt v2_scaling_r1_build -S -5000 > /tmp/v2_r1_final.log`）

## 信息引用

- 现场证据：用户目视检查 Stage 3 meshes 和 Stage 4 渲染结果
- VLM review 佐证：obj_067 overexposed (score 0.55), obj_068 color_shift (0.53), obj_069 underexposed (0.41), obj_070 flat_lighting (0.57)
- 文件大小证据：见本文件上方表格
- 之前提示文件：同目录 `HINT_FOR_CODEX_20260421.md`（请以**本文件**为准，之前的诊断不完整）

---

**给 codex 的一句话摘要**：
> Stage 3 Hunyuan3D 在 obj_055 之后的物体上 texture baking 失败（整批 mesh 白色），另外 7 个物体（051/054/057/060/063/066/069）连几何体贴图都没有。**不要继续 Phase 6，先诊断 Stage 3 再决定是重跑还是改方案**。
