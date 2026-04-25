# Stage 3 Texture Baking 诊断与修复 — v2 Scaling R1

你是 v2 Scaling R1 数据构建的接管 agent。当前 Phase 5（rotation export）已完成，但用户发现 **obj_055 之后的所有物体 3D 重建结果是白色（缺失颜色）**。你的任务是诊断并修复这个问题。

**IMPORTANT**: 本机是 Windows (PowerShell)。所有服务器操作通过 SSH。

---

## 核心问题

从 obj_055 开始，后面所有物体的 3D mesh（Stage 3 Hunyuan3D 输出）是白色/缺色。

### Mesh 文件大小证据

```
obj_051: 1.2 MB  ← 无贴图（每3个一次，疑似某 GPU worker texture baking 全失败）
obj_052: 32.5 MB ← 有贴图
obj_053: 14.2 MB ← 有贴图
obj_054: 1.3 MB  ← 无贴图
obj_055: 12.4 MB ← 有大小但用户说白色
obj_056: 22.9 MB
obj_057: 1.4 MB  ← 无贴图
obj_058: 22.5 MB
obj_059: 20.0 MB
obj_060: 1.5 MB  ← 无贴图
obj_061: 10.7 MB
obj_062: 3.5 MB
obj_063: 1.3 MB  ← 无贴图
obj_064: 11.4 MB
obj_065: 15.4 MB
obj_066: 1.4 MB  ← 无贴图
obj_067: 6.6 MB
obj_068: 6.8 MB
obj_069: 1.3 MB  ← 无贴图
obj_070: 12.8 MB
```

**两个问题**：
1. **7 个物体（051/054/057/060/063/066/069）只有 ~1.2-1.5 MB** — 纯几何，完全没有 texture。每 3 个一次，说明 3 GPU 并行时某个 GPU 的 Hunyuan3D texture baking 一直失败。
2. **obj_055+ 即使文件大也渲染成白色** — 可能 albedo/baseColor 贴图数据异常。

---

## 服务器信息

| 服务器 | SSH | GPU | 用途 |
|--------|-----|-----|------|
| wwz | `wwz` | 3×A800 | 数据构建（Stage 3 Hunyuan3D 在此运行） |
| 68 | `zhanghy56_68` | 8×H100 | 训练 + 评测 |

- wwz 代码根: `/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code`
- wwz Python: `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`
- wwz Blender: `/home/wuwenzhuo/blender-4.24/blender`
- wwz 必须用 tmux（screen 不可用）
- 跨服务器传输经本地中继

### 关键路径

- Stage 1-3 产物: `$WWZ_CODE/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/`
- Meshes: `$WWZ_CODE/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/meshes/`
- VLM loop 产物: `$WWZ_CODE/pipeline/data/evolution_v2_scaling_r1_feedback_objects_20260421/`
- Rotation export: `$WWZ_CODE/pipeline/data/v2_scaling_r1_rotation_export_20260421/`
- Stage 3 脚本: `$WWZ_CODE/pipeline/stage3_image_to_3d.py`
- Stage 2.5 脚本: `$WWZ_CODE/pipeline/stage2_5_sam2_segment.py`（实际用 SAM3）
- Hunyuan3D 相关代码: 检查 `$WWZ_CODE/pipeline/` 下的 Stage 3 相关文件
- SAM3 模型: `/huggingface/model_hub/sam3/sam3.pt`
- SAM3 库: `/aaaidata/zhangqisong/data_build/sam3`

---

## 你的任务

### Step 1: 诊断（~10 min）

1. **查 Stage 3 日志**：找出 texture baking 为什么失败
```bash
ssh wwz "find /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421 -name '*.log' -o -name '*stage3*' -o -name '*hunyuan*' 2>/dev/null | head -20"
```

2. **用 python 检查 mesh material**：确认"大文件"的 mesh 是否真的有有效贴图
```bash
ssh wwz "/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c \"
import trimesh
for obj_id in ['obj_052', 'obj_055', 'obj_060', 'obj_065']:
    path = f'/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/meshes/{obj_id}.glb'
    try:
        m = trimesh.load(path)
        if isinstance(m, trimesh.Scene):
            for name, g in m.geometry.items():
                mat = getattr(g.visual, 'material', None)
                has_tex = getattr(mat, 'baseColorTexture', None) is not None if mat else False
                print(f'{obj_id}: geometry={name} material={type(mat).__name__} has_texture={has_tex}')
        else:
            print(f'{obj_id}: single mesh, visual={type(m.visual).__name__}')
    except Exception as e:
        print(f'{obj_id}: ERROR {e}')
\""
```

3. **查 Stage 3 脚本的 GPU 分配逻辑**：确认 3 GPU 怎么分配的
```bash
ssh wwz "head -100 /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/stage3_image_to_3d.py"
```

4. **查 T2I 白底图（Stage 2 输出）**：确认输入图是正常的，问题出在 Stage 3
```bash
ssh wwz "ls /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/t2i/ 2>/dev/null | head -20"
# 查看白底图文件大小
ssh wwz "ls -la /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/t2i/obj_054*.png /aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code/pipeline/data/dataset_v2_scaling_r1_feedback_stage1_assets_20260421/t2i/obj_055*.png 2>/dev/null"
```

### Step 2: 确定修复方案

根据诊断结果选择：

**方案 A（推荐，如果只是 Hunyuan3D GPU 问题）**：
- 修复 Stage 3 脚本的 GPU 分配 / texture baking 参数
- 只重跑失败的物体（需要确定具体哪些是坏的）
- 保留 Stage 1 和 Stage 2 产物（T2I 白底图 + SAM3 mask）

**方案 B（如果是批量系统问题）**：
- 整批 20 物体重跑 Stage 3
- 保留 Stage 1-2 产物

**方案 C（如果 Stage 3 问题太深）**：
- 只保留 obj_051-054 中有效的物体
- 将预算缩减到 4-5 个新物体先完成一次 R1 验证

### Step 3: 执行修复

修复完成后：
1. 验证新 mesh 有颜色（非白色）
2. 重跑 VLM loop bootstrap（只对重新生成 mesh 的物体）
3. 重跑 rotation export
4. 回报给我完整状态

---

## 关键约束

- **不要**覆盖原有产物目录，新建目录
- **不要**传坏数据到 68
- **不要**修改 `stage4_blender_render.py`（废弃脚本）
- **不要**删除 `evolution_v2_scaling_r1_feedback_objects_20260421/`（保留做诊断）
- 渲染用物体旋转（`yaw_deg`），不是相机轨道
- wwz 用 tmux，不用 screen
- 保留所有日志

## 输出报告

完成后报告：
1. Stage 3 失败根因
2. 哪些物体需要重新生成（具体 ID 列表）
3. 修复方案和执行结果
4. 新的 pipeline 进度和下一步
