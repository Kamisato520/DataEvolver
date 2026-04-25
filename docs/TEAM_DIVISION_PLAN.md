# 团队分工方案：内部 Loop + 外部反馈数据构建

## 一、系统架构总览

**约束**：两名成员都在 **wwz 服务器**（3×A800）上工作，只负责**数据集构建**。68 服务器上的训练、评测、compare 由你亲自操作。Bench 评测结果由你提供给他们作为下一轮的输入。

```
┌──────────────────────────────────────────────────────────────────────┐
│                        你（总指挥）                                   │
│                                                                      │
│  · 68 服务器：训练 + 评测 + compare                                   │
│  · 产出 bench 结果 → 传递给 A/B                                      │
│  · 决定弱角度、物体预算、门控阈值                                       │
│  · wwz→68 数据传输                                                    │
│  · 最终 verdict 判定 + 策略调整                                        │
└──────┬──────────────────────────────────────────────┬────────────────┘
       │ 弱角度列表 + 物体预算                          │ trainready 数据
       │ bench 结果 + 逐物体分析                        │ (wwz→68 传输)
       ▼                                               ▲
┌──────────────────────────────────────────────────────────────────────┐
│                    wwz 服务器（3×A800）                               │
│                                                                      │
│  ┌─────────────────────┐         ┌─────────────────────────────┐    │
│  │  人员 A：内部 Loop    │ ──→──→ │  人员 B：外部反馈数据构建     │    │
│  │  （渲染质量优化）     │ 高质量  │  （Pipeline 编排 + 交付）    │    │
│  │                      │ 物体    │                              │    │
│  │  VLM 评审 → 参数调优  │        │  概念生成 → Stage 1-3        │    │
│  │  → 再渲染 → 门控     │ ←──←── │  → 交给 A 做 VLM loop       │    │
│  │                      │ mesh    │  → 收 A 的门控结果           │    │
│  │  产出：高质量渲染     │        │  → Rotation Export           │    │
│  │  + gate_report       │        │  → Trainready → 交付给你     │    │
│  └─────────────────────┘         └─────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

**A 和 B 的协作关系**：B 是 pipeline 的"总包"，负责从概念到交付的全流程编排；A 是"质量专家"，专注于 VLM 渲染优化这个最难的环节。B 把 Stage 3 产出的 mesh 交给 A 做 VLM loop，A 把优化好且通过门控的物体交回 B 做后续 export 和打包。

---

## 二、人员 A：内部 Loop（渲染质量优化）

### 2.1 一句话定位

**让每个物体的 Blender 渲染质量尽可能高**——这是 R1 失败的根因（hybrid_score 全部 < 0.6），也是 R2 最关键的改进点。

### 2.2 负责的子系统

| 子系统 | 说明 | 关键脚本 |
|--------|------|---------|
| VLM Render Loop | Blender 渲染 → Qwen3.5-VL 评审 → 参数调优 → 再渲染 | `scripts/bootstrap_scene_yaw000_objects.py` |
| VLM 评审 | 5 维度打分（lighting/material/camera/composition/realism） | `pipeline/stage5_5_vlm_review.py` |
| 参数调优 | VLM 建议 → Blender 参数修改 | `pipeline/stage5_6_feedback_apply.py` |
| Golden Config Prior | 高质量渲染配置聚合，用于 warm-start | `scripts/feedback_loop/build_render_prior_library.py` |
| 质量门控 | 按 hybrid_score 过滤物体 | `scripts/feedback_loop/quality_gate.py`（待开发） |

### 2.3 R2 具体任务

| 优先级 | 任务 | 说明 | 预计工时 |
|--------|------|------|---------|
| **P0** | **开发 quality_gate.py** | 读取 VLM bootstrap 输出，按 hybrid_score 过滤，输出 gate_report.json | 3h |
| **P0** | **VLM bootstrap 参数优化** | max_rounds 10→15, plateau_window 2→3, plateau_eps 0.01→0.008 | 1h |
| **P1** | **改进 golden config prior** | 用 baseline top-20 物体配置重新聚合，或加入 R1 中最好的物体配置 | 2h |
| **P1** | **VLM 评审维度分析** | 分析 5 个维度中哪些与最终训练效果相关性最高，调整权重 | 3h |
| **P2** | **自适应收敛策略** | 根据分数趋势动态决定是否继续迭代（替代固定 plateau_window） | 4h |

### 2.4 工作流程

```
收到 B 交付的 mesh（GLB 文件）
    ↓
为每个物体执行 VLM Render Loop：
    ① Blender 加载 mesh → 用 golden config prior 初始化参数 → 渲染
    ② Qwen3.5-VL 从 5 维度评审 → 输出 hybrid_score + 改进建议
    ③ stage5_6 将建议转为 Blender 参数调整
    ④ 重复 ①-③ 直到：score 收敛 / 达到 max_rounds / score ≥ 阈值
    ↓
质量门控：
    · hybrid_score ≥ 0.55 → 通过，交给 B
    · hybrid_score < 0.55 → 拒绝，记入 rejected_objects.json
    ↓
输出：
    · 每个通过物体的最佳 control_state（JSON）
    · gate_report.json（通过率、分数分布）
    · rejected_objects.json（拒绝原因）
```

### 2.5 交付物（给 B）

| 交付物 | 格式 | 说明 |
|--------|------|------|
| 每个通过物体的 `best_state.json` | JSON | 包含最佳渲染参数 |
| `gate_report.json` | JSON | 通过/拒绝数、分数分布、通过率 |
| `rejected_objects.json` | JSON | 被拒物体 + hybrid_score + 原因 |

### 2.6 质量 KPI

| 指标 | R1 实际值 | R2 目标 |
|------|----------|--------|
| 通过门控的物体比例 | 0%（无门控，全部 < 0.6） | ≥ 50%（20/40） |
| 通过物体 hybrid_score 均值 | — | ≥ 0.60 |
| VLM bootstrap 平均轮次 | ~3（plateau_window=2 过早停） | 6-10 |
| hybrid_score 最低通过值 | — | ≥ 0.55 |

### 2.7 需要了解的关键背景

1. **hybrid_score 构成**：VLM 5 维度评审的加权平均，0-1 范围。>0.78 高质量，0.55-0.78 可用，<0.55 噪声
2. **R1 失败原因**：20 个新物体 hybrid_score 最高仅 0.582（obj_052），最低 0.368（obj_059），全部低于门控线。导致训练数据中 20% 是低质量噪声，DINO 在 4 个角度退化
3. **plateau_window=2 的问题**：连续 2 轮 score 变化 < 0.01 就停止迭代。很多物体在第 2-3 轮就停了，没有充分优化
4. **Golden config prior**：设计门槛 0.78 实际筛不出数据，被迫下调到 0.58 得到 58 条 prior。prior 质量偏低可能导致 warm-start 效果有限
5. **Blender 参数空间**：lighting（HDRI、强度、角度）、material（粗糙度、金属度）、camera（距离、FOV、elevation）、scene（背景色、地面反射）、object（缩放、位置偏移）

---

## 三、人员 B：外部反馈数据构建（Pipeline 编排 + 交付）

### 3.1 一句话定位

**编排从概念生成到 trainready 交付的全流程 pipeline**，确保 Stage 1-3 稳定运行、与 A 高效协作、最终交付可直接用于训练的数据集。

### 3.2 负责的子系统

| 子系统 | 说明 | 关键脚本 |
|--------|------|---------|
| 新物体概念生成 | 去重 + category 分布采样 | `scripts/feedback_loop/build_feedback_expansion_objects.py` |
| Stage 1: Text Expansion | 种子概念 → 详细文字描述 | `pipeline/stage1_text_expansion.py` |
| Stage 2: T2I Generate | prompt → 白底参考图 1024×1024 | `pipeline/stage2_t2i_generate.py` |
| Stage 2.5: SAM3 Segment | 白底图 → RGBA mask | （SAM3 pipeline） |
| Stage 3: Hunyuan3D | 白底图 + mask → GLB mesh | `pipeline/stage3_image_to_3d.py` |
| Rotation Export | 最佳 state → 多角度 Blender 渲染 | `scripts/export_rotation8_from_best_object_state.py` |
| Trainready Build | 渲染图 → JSONL pairs + views/ | `scripts/build_rotation8_trainready_dataset.py` |
| Pipeline 编排 | 全流程串联 + 错误处理 | shell 脚本（待开发/完善） |

### 3.3 R2 具体任务

| 优先级 | 任务 | 说明 | 预计工时 |
|--------|------|------|---------|
| **P0** | **修复 Stage 3 Paint 静默失败** | `stage3_image_to_3d.py` 中 Paint 失败改为抛异常 | 2h |
| **P0** | **Stage 3 单 GPU 策略落地** | GPU 0 单卡串行，或写逐 GPU smoke test 脚本 | 1h |
| **P1** | **新物体概念生成（40 个）** | obj_071~110，去重 baseline 50 + R1 的 20 | 2h |
| **P1** | **端到端 pipeline 脚本** | 封装 Stage 1→2→2.5→3→(交给A)→export→trainready 为一键脚本 | 4h |
| **P1** | **Rotation export 集成门控** | 增加 `--min-hybrid-score` 参数，或只接受 A 的通过列表 | 1h |
| **P2** | **Stage 1 输出质量检查** | 验证 T2I 白底图质量（无背景噪点、物体居中、分辨率正确） | 2h |
| **P2** | **manifest 生成完善** | trainready 的 manifest.json 包含质量元数据 | 1h |

### 3.4 工作流程

```
收到你提供的信息：
    · 弱角度列表（如 90°/180°/270°）
    · 新物体预算（如 40 个）
    · 质量门控阈值（如 hybrid_score ≥ 0.55）
    ↓
概念生成：
    build_feedback_expansion_objects.py → feedback_expansion_r2_objects.json
    ↓
Stage 1: Text Expansion
    种子概念 → 详细 prompt
    ↓
Stage 2: T2I + SAM3
    prompt → 白底图 1024×1024 → RGBA mask
    ↓
Stage 3: Hunyuan3D（GPU 0 单卡）
    白底图 + mask → GLB mesh（必须有贴图）
    验证：GLB > 5MB = 有贴图 ✅ / < 1MB = 无贴图 ❌
    ↓
===== 交接给 A =====
    把 mesh 交给 A 做 VLM Render Loop
    等待 A 返回 gate_report + 通过物体的 best_state
===== 从 A 接收 =====
    ↓
Rotation Export（仅通过门控的物体）
    best_state → yaw000 + 弱角度渲染
    ↓
Trainready Build
    渲染图 → JSONL pairs + views/ + manifest.json
    ↓
最终交付给你
    trainready/ 目录（含 gate_report.json）
```

### 3.5 交付物（给你）

| 交付物 | 格式 | 说明 |
|--------|------|------|
| `trainready/pairs/train_pairs.jsonl` | JSONL | 通过门控的训练 pairs |
| `trainready/views/` | PNG 目录 | 渲染原图（source + target） |
| `trainready/objects/` | 目录 | 物体元数据 |
| `trainready/manifest.json` | JSON | pair 数量、物体数、角度覆盖 |
| `gate_report.json` | JSON | 来自 A 的质量门控报告 |

### 3.6 质量 KPI

| 指标 | R1 实际值 | R2 目标 |
|------|----------|--------|
| Stage 3 Paint 成功率 | 35%（7/20，首次 3GPU） | 100%（单 GPU） |
| T2I 白底图合格率 | ~100% | 100% |
| GLB 有贴图率 | 100%（修复后） | 100% |
| pipeline 端到端成功率 | ~70%（多处手动干预） | ≥ 95% |
| 交付周期（40 物体） | — | ≤ 8h（不含 VLM loop 等待） |

### 3.7 需要了解的关键背景

1. **Stage 3 多 GPU 坑**：Hunyuan3D 的 Paint 管线依赖 CUDA extension（custom_rasterizer / DifferentiableRenderer），只在 GPU 0 上编译兼容。GPU 1/2 会静默 fallback 到无贴图 mesh（shape-only GLB < 1MB），不报错。R1 中 13/20 物体因此需要 GPU 0 重跑
2. **白底图要求**：Stage 1 的 T2I 输出必须是白色背景，非白底会影响后续 SAM3 分割和 Hunyuan3D 重建质量
3. **trainready 路径约束**：`source_image` / `target_image` 必须指向 `views/` 下的渲染原图（不是 `bbox_views/`）
4. **弱角度**：由你从 SpatialEdit-Bench 评测结果中确定，R1 是 90°/180°/270°。R2 可能沿用或调整
5. **Rotation Export 只导出 yaw000 + 弱角度**：不是全 7 角度。每物体 4 张渲染（1 源 + 3 目标），每物体生成 3 个 pair

---

## 四、A 与 B 的接口契约

### 4.1 B → A（mesh 交接）

```
交接目录：pipeline/data/dataset_v2_scaling_r2_stage3_output/
├── obj_071/
│   ├── mesh.glb          # 有贴图的 GLB（> 5MB）
│   ├── reference.png     # 白底参考图
│   └── metadata.json     # {obj_id, category, stage3_gpu, glb_size_mb}
├── obj_072/
│   └── ...
└── handoff_manifest.json  # {total: 40, paint_success: 40, ready_for_vlm: true}
```

**B 的交付标准**：
- 每个 GLB 必须有贴图（文件 > 5MB）
- 提供白底参考图供 A 在 VLM 评审中参考
- handoff_manifest.json 确认数量

### 4.2 A → B（门控结果交接）

```
交接目录：pipeline/data/v2_scaling_r2_vlm_gated_output/
├── passed/
│   ├── obj_071/
│   │   ├── best_state.json     # 最佳渲染参数
│   │   └── best_render.png     # 最佳渲染结果（yaw000 方向）
│   ├── obj_073/
│   └── ...
├── gate_report.json            # {passed: 22, rejected: 18, pass_rate: 55%, ...}
└── rejected_objects.json       # [{obj_id, hybrid_score, reason}, ...]
```

**A 的交付标准**：
- 通过物体的 hybrid_score ≥ 门控阈值（你指定）
- gate_report 包含分数分布（min/max/mean/median）
- rejected_objects 记录拒绝原因

### 4.3 协作时序

```
B: Stage 1-2-2.5-3（~2h）
    ↓ mesh 交给 A
A: VLM Render Loop（~3-5h，3×A800 并行）
    ↓ 门控结果交给 B
B: Rotation Export + Trainready（~1h）
    ↓ trainready/ 交给你
你: wwz→68 传输 → 合并 → 训练 → 评测 → compare
```

---

## 五、协作时间线（R2）

### Day 1：准备阶段（并行）

| 时段 | 你 | A | B |
|------|------|------|------|
| 上午 | R1 逐物体分析（68 上 per_pair.csv），确定门控阈值/物体预算 | 开发 quality_gate.py | 修复 Stage 3 Paint 异常 |
| 下午 | 将弱角度列表 + 预算 + 阈值告知 A/B | VLM bootstrap 参数优化 + golden config 更新 | Stage 3 单 GPU 策略 + 概念生成（40 物体） |
| 晚上 | review 两人代码 | — | — |

### Day 2：数据构建（A/B 协作）

| 时段 | 你 | A | B |
|------|------|------|------|
| 上午 | — | 等待 B 的 mesh | Stage 1→2→2.5→3 执行（~2h） |
| 中午 | — | 收到 mesh，启动 VLM loop（3×A800） | 编写端到端 pipeline 脚本 |
| 下午~晚上 | — | VLM loop 运行中（3-5h） | 等待 A 的门控结果 |

### Day 3：交付 + 训练

| 时段 | 你 | A | B |
|------|------|------|------|
| 上午 | — | 门控完成，交付给 B | 收到门控结果，Rotation Export + Trainready（~1h） |
| 中午 | 收到 B 的 trainready，wwz→68 传输 | 分析 VLM loop 日志，总结改进效果 | — |
| 下午 | 68 上合并 augmented dataset + 验证 + 启动训练 | — | — |
| 过夜 | 训练运行中（~10h） | — | — |

### Day 4：评测 + 决策

| 时段 | 你 | A | B |
|------|------|------|------|
| 上午 | 6-GPU 评测 + compare | — | — |
| 下午 | 结果分析 + verdict + 决定是否 R3 | — | — |

---

## 六、风险与应对

| 风险 | 影响 | 负责人 | 应对 |
|------|------|--------|------|
| 门控后通过物体太少（< 15） | 数据不足 | A + 你 | 降阈值到 0.50 或追加物体预算 |
| Stage 3 GPU 0 也出 Paint 问题 | 全部阻塞 | B | 检查 CUDA extension 编译状态，必要时重编译 |
| VLM bootstrap 仍收敛太快 | 质量不够 | A | 继续增大 max_rounds 到 20 |
| wwz GPU 资源被占 | 数据构建延迟 | B | tmux 抢占 + 协调 wwz 其他用户 |
| A/B 交接延迟 | 总时间拉长 | 你 | 明确交接时间点，设 deadline |

---

## 七、各角色手册速查

### 你的操作清单

1. 68 上跑 R1 per_pair.csv 逐物体分析 → 确定门控阈值
2. 告知 A/B：弱角度列表、物体预算、门控阈值
3. Review A/B 的代码改动
4. 收到 B 的 trainready/ 后：
   - `ssh wwz "tar czf - -C <dir> ." | ssh zhanghy56_68 "tar xzf - -C <dest>"`
   - 68 上 `build_augmented_rotation_dataset.py` + `validate_dataset.py`
   - 启动训练 → 评测 → compare
5. 根据 verdict 决定下一步

### A 的操作清单

1. 开发 quality_gate.py（本地或 wwz 上）
2. 修改 `bootstrap_scene_yaw000_objects.py` 的默认参数
3. 更新 golden config prior
4. 收到 B 的 mesh 后，启动 VLM bootstrap：
   ```bash
   tmux new-session -s vlm_loop
   python3 bootstrap_scene_yaw000_objects.py \
     --objects-dir <mesh_dir> \
     --output-dir <evolution_dir> \
     --max-rounds 15 --plateau-window 3 \
     --render-prior-library golden_config_library.json
   ```
5. VLM loop 完成后，运行 quality_gate.py
6. 将门控结果交给 B

### B 的操作清单

1. 修复 `stage3_image_to_3d.py` 的 Paint 异常处理
2. 确认 GPU 0 单卡策略
3. 生成新物体概念（40 个）
4. 执行 Stage 1→2→2.5→3：
   ```bash
   tmux new-session -s pipeline
   # Stage 1
   python3 pipeline/stage1_text_expansion.py --concepts feedback_expansion_r2_objects.json --output-dir <stage1_out>
   # Stage 2
   python3 pipeline/stage2_t2i_generate.py --prompts <stage1_out>/prompts.json --output-dir <stage2_out>
   # Stage 2.5: SAM3
   # Stage 3 (GPU 0 only)
   CUDA_VISIBLE_DEVICES=0 python3 pipeline/stage3_image_to_3d.py --input-dir <stage2_out> --output-dir <stage3_out>
   ```
5. 将 mesh 交给 A
6. 收到 A 的门控结果后：
   ```bash
   # Rotation Export
   python3 scripts/export_rotation8_from_best_object_state.py \
     --evolution-dir <vlm_gated_output>/passed \
     --rotations 0,90,180,270 \
     --output-dir <export_dir>
   # Trainready
   python3 scripts/build_rotation8_trainready_dataset.py \
     --renders-dir <export_dir> \
     --target-rotations 90,180,270 \
     --prompt-version v3 \
     --output-dir <trainready_dir>
   ```
7. 将 trainready/ 交给你

---

## 八、wwz 服务器环境速查

```bash
# SSH
ssh wwz

# 代码目录
WWZ_CODE=/aaaidata/zhangqisong/data_build_runs/scene_full20_loop_20260404_code
cd $WWZ_CODE

# Python 环境
export PYTHON=/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3

# 必须用 tmux（screen 不可用）
tmux new-session -s <session_name>

# GPU 状态
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 关键目录
ls pipeline/data/          # 所有数据产出
ls configs/seed_concepts/  # 物体概念定义
ls pipeline/               # Stage 脚本
ls scripts/                # 工具脚本
```

---

## 九、一句话总结

> **A 专攻 VLM 渲染质量（让 hybrid_score 上去），B 编排数据 pipeline（让 Stage 1-3 稳定跑通），你做决策和 68 训练评测。三人协作，A/B 全部在 wwz 上工作，预计 4 天跑完 R2。**
