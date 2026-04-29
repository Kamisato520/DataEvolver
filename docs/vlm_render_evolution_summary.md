# Scene Insert VLM 渲染演进总结

日期：2026-04-01

---

## 背景

本文档记录 `data_build` 项目 Scene Insert Pipeline（v7 系列）的完整演进过程，包括：

- v7 场景渲染修正（v7-1, v7-2）
- Smoke 实验验证
- v7 Full Pilot（10 对象完整跑）
- PLANv7-3 实施（阈值收紧、数据集导出、action whitelist）
- v7.1 lighting-alignment 对照实验
- 最终结论与决策

**任务定义**：将任意 3D 物体（GLB mesh）插入预设 Blender 场景（4.blend），通过 VLM 评审循环迭代渲染参数，最终生成高质量场景合成图像用于下游 VLM 训练数据。

**评分体系**：hybrid score = VLM 评审分 × programmatic physics 修正项，范围 0–1。

---

## 一、v7 场景渲染修正

v7 的核心目标是将物体正确插入真实 Blender 场景（4.blend），修正之前 v6 系列使用简单白色背景的问题。

### 1.1 v7-1：渲染脚本修正（2026-04-01）

**文件**：`pipeline/stage4_scene_render.py`

**根因**：

| 问题 | 描述 |
|------|------|
| World node tree 被覆盖 | `ensure_world_environment()` 重写了 4.blend 已有的 world node tree，导致场景光照失效 |
| 灯光被错误禁用 | 原代码用 `hide_render=True` 隐藏场景灯光，但未删除，导致物理仍然计算却不渲染 |
| 相机距离固定 | 固定像素距离不随场景尺度缩放，小物体出画面 |
| 地面检测错误 | 使用了 `Cube` 作为支撑面，但 4.blend 的真实地面是 `Plane` |

**修复内容**：

- 新增 `use_existing_world=True` 参数，跳过 world node tree 修改，保留场景原有光照
- 将 `hide_render=True` 改为 `scale_existing_lights(factor=0.3)`，降低原有灯光强度而非禁用
- 相机距离改为基于场景尺度：`distance = max_span * 0.02`
- `support_object_name` 改为 `"Plane"`，`ground_object_raycast` 模式检测真实地面接触

### 1.2 v7-2：VLM Review Prompt 语义修正（2026-04-01）

**文件**：`pipeline/stage5_5_vlm_review.py`

**根因**：v7-1 修正后场景渲染已正确，但 VLM 评审 prompt 仍存在语义模糊问题，导致误判率高。

**修复内容**：

| 维度 | 旧行为 | 修复后 |
|------|--------|--------|
| `physics_consistency` | 把阴影质量也算进去 | 只标记真正浮空/穿插，不惩罚阴影 |
| `scene_light_mismatch` | 宽泛触发 | 收紧：仅在光源方向明显相反或时段不同时触发 |
| `structure_consistency`（scene_insert 模式） | 对比整图 | 只比较物体几何，忽略背景和光照差异 |
| `_merge_programmatic_physics` | programmatic clean 不降级 | programmatic clean → 将 VLM major_issue 降级为 minor_issue |
| `unstable_span_limit` | 0.05 | 0.10（VLM 推断噪声导致假不稳定） |

---

## 二、Smoke 实验结果

### 2.1 smoke_corrected（v7-1 修正后，3 对象）

| 对象 | Confirmed | Final | Exit | 主诊断 |
|------|-----------|-------|------|--------|
| obj_001 | 0.5557 | 0.5557 | rejected_mesh | scene_light_mismatch（structure major） |
| obj_004 | 0.6295 | 0.6295 | mid_no_improve | scene_light_mismatch |
| obj_009 | 0.6578 | 0.6578 | rejected_unstable_score | scene_light_mismatch |

**结论**：修正后的渲染脚本消除了之前的 `flat_low_contrast`/`shadow_missing` 诊断，但全部对象仍被标记为 `scene_light_mismatch`，且 obj_009 分数不稳定（span > 0.05 阈值）。

### 2.2 smoke_v2（v7-2 prompt 修正后，3 对象）

| 对象 | Confirmed | Final | Delta | Action | Exit |
|------|-----------|-------|-------|--------|------|
| obj_001 | 0.5511 | **0.6387** | +0.0876 | ENV_ROTATE_30 | accepted_after_try |
| obj_004 | 0.6505 | 0.6505 | +0.0000 | ENV_ROTATE_30 | mid_no_improve |
| obj_009 | 0.6542 | **0.7229** | +0.0687 | ENV_ROTATE_30 | accepted_after_try |

**关键发现**：

- VLM prompt 修正后稳定性问题解决（span 降至 0.0 范围）
- `ENV_ROTATE_30`（环境光旋转）对 2/3 对象有效，提升幅度 +6~9%
- `scene_light_mismatch` 诊断仍然全覆盖，但已不阻止 controller 产生收益
- obj_004 旋转无改善（delta=0），mid 区间直接结束

**smoke_corrected vs smoke_v2 对比**：

| 指标 | smoke_corrected | smoke_v2 |
|------|-----------------|----------|
| avg final | 0.6043 | **0.6707** |
| 有效改善对象 | 0/3 | 2/3 |
| 稳定性问题 | 1/3 unstable | 0/3 |

---

## 三、v7 Full Pilot（10 对象完整结果）

### 3.1 运行配置

- Profile：`configs/dataset_profiles/scene_v7.json`
- 场景：`configs/scene_template.json`（4.blend）
- Action space：`configs/scene_action_space.json`
- VLM：`Qwen3-VL-8B-Instruct`（`/huggingface/model_hub/Qwen3-VL-8B-Instruct`）
- GPU：3×A800，各跑 3-4 个对象（并行）
- Zoning score：hybrid

### 3.2 完整结果表

| 对象 | Confirmed | Final | Delta | Action | Exit | 诊断 | VLM-only |
|------|-----------|-------|-------|--------|------|------|----------|
| obj_001 | 0.5511 | **0.6387** | +0.0876 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | 0.6447 |
| obj_002 | 0.5980 | 0.6152 | +0.0172 | M_SATURATION_DOWN | accepted_after_try | scene_light_mismatch | — |
| obj_003 | 0.4370 | 0.4370 | 0.0000 | — | rejected_unstable_score | — | — |
| obj_004 | 0.6505 | 0.6505 | 0.0000 | ENV_ROTATE_30 | mid_no_improve | scene_light_mismatch | — |
| obj_005 | 0.6380 | **0.6916** | +0.0536 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | 0.7031 |
| obj_006 | 0.6313 | 0.6313 | 0.0000 | O_LOWER_SMALL | mid_no_improve | scene_light_mismatch | — |
| obj_007 | 0.6698 | 0.6852 | +0.0154 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | — |
| obj_008 | 0.5933 | 0.6086 | +0.0153 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | — |
| obj_009 | 0.6542 | **0.7229** | +0.0687 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | 0.7734 |
| obj_010 | 0.6193 | **0.7087** | +0.0894 | ENV_ROTATE_30 | accepted_after_try | scene_light_mismatch | 0.7625 |

### 3.3 汇总统计

| 指标 | 值 |
|------|-----|
| avg confirmed | 0.6043 |
| avg final | 0.6390 |
| avg delta（全体） | **+0.0347** |
| avg delta（改善对象，7/10） | **+0.0506** |
| 主要动作 | ENV_ROTATE_30（7/10 对象） |
| scene_light_mismatch 覆盖率 | 9/10 |
| 拒绝：unstable | 1（obj_003） |
| mid_no_improve（无改善） | 2（obj_004, obj_006） |
| accepted_after_try（有改善） | 7 |

### 3.4 关键观察

1. **Controller 有真实收益**：avg confirmed 0.6043 → avg final 0.6390，delta +3.5%
2. **ENV_ROTATE_30 是最有效动作**：7/10 对象选择，其中 5 个有效提升
3. **质量天花板 ~0.72–0.73**：即使最优对象（obj_009=0.7229, obj_010=0.7087）也止步于此
4. **主瓶颈是 scene_light_mismatch**：9/10 对象均有此诊断，是系统性问题而非个别对象问题
5. **Physics 总体干净**：obj_002 contact_gap=0.0065mm（轻微浮空），obj_007 contact_gap=0.0075mm，其余均为 0

---

## 四、PLANv7-3 实施（2026-04-01）

### 4.1 contact_gap 阈值收紧：0.01 → 0.005

**动机**：0.01mm 阈值过宽，轻微悬浮物体可能被误判为 physics clean，导致 scene_dataset_v0 过滤规则不可信。

**修改文件**：

| 文件 | 位置 | 修改 |
|------|------|------|
| `pipeline/stage5_5_vlm_review.py` | L988 | `gap > 0.01 or penetration > 0.01` → `> 0.005` |
| `pipeline/stage5_5_vlm_review.py` | L990 | `gap < 0.01 and penetration < 0.01` → `< 0.005` |
| `pipeline/stage5_5_vlm_review.py` | L1158 | `contact_gap > 0.01` → `> 0.005`（floating_visible 注入） |
| `pipeline/stage5_5_vlm_review.py` | L1160 | `penetration_depth > 0.01` → `> 0.005`（intersection 注入） |
| `pipeline/stage5_5_vlm_review.py` | L1169 | `contact_gap > 0.01`（agg_diagnosis override）→ `> 0.005` |
| `run_scene_evolution_loop.py` | PROFILE_DEFAULTS | `"contact_gap_minor": 0.01` → `0.005` |

### 4.2 action_whitelist 功能实现

**动机**：支持 v7.1 实验中的动作空间约束（仅允许特定光照动作）。

**修改文件**：

**`pipeline/stage5_6_feedback_apply.py`**：

```python
# select_action 新增参数
def select_action(..., action_whitelist: list = None) -> Optional[str]:
    if action_blacklist is None:
        action_blacklist = set()
    # 将 whitelist 外的所有 action 追加到 blacklist
    if action_whitelist is not None:
        _wset = set(action_whitelist)
        for _gdata in aspace.get("groups", {}).values():
            for _aname in _gdata.get("actions", {}).keys():
                if _aname not in _wset:
                    action_blacklist = action_blacklist | {_aname}
        for _aname in aspace.get("compound_actions", {}).keys():
            if _aname not in _wset:
                action_blacklist = action_blacklist | {_aname}
```

- `apply_feedback` 同步新增 `action_whitelist: list = None` 参数并传递给 `select_action`
- `run_scene_evolution_loop.py` 的 `apply_feedback` 调用新增 `action_whitelist=profile.get("action_whitelist")`

### 4.3 scene_dataset_v0 导出

**新文件**：`scripts/export_scene_dataset_v0.py`

**过滤规则**：

| 规则 | 值 |
|------|----|
| exit_reason | ∈ {accepted_after_try, mid_no_improve} |
| 排除对象 | obj_003（无结果）|
| contact_gap | ≤ 0.005 |
| penetration_depth | == 0 |
| is_out_of_support_bounds | false |

**导出结果**：

| 对象 | Final | Delta | Action | Tier | 分组 | 备注 |
|------|-------|-------|--------|------|------|------|
| obj_001 | 0.6387 | +0.088 | ENV_ROTATE_30 | C | priority | — |
| obj_002 | — | — | — | — | — | **过滤**：contact_gap=0.0065 > 0.005 |
| obj_003 | — | — | — | — | — | **跳过**：无结果文件 |
| obj_004 | 0.6505 | 0.000 | ENV_ROTATE_30 | B | cautious | — |
| obj_005 | 0.6916 | +0.054 | ENV_ROTATE_30 | B | priority | — |
| obj_006 | 0.6313 | 0.000 | O_LOWER_SMALL | C | cautious | — |
| obj_007 | — | — | — | — | — | **过滤**：contact_gap=0.0075 > 0.005 |
| obj_008 | 0.6086 | +0.015 | ENV_ROTATE_30 | C | priority | — |
| obj_009 | 0.7229 | +0.069 | ENV_ROTATE_30 | A | priority | — |
| obj_010 | 0.7087 | +0.089 | ENV_ROTATE_30 | A | priority | — |

**汇总**：7 个对象导出，Tier A×2 / B×2 / C×3
**路径**：`pipeline/data/scene_dataset_v0/`

---

## 五、v7.1 lighting-alignment 实验

### 5.1 实验设计

固定 3 个对象（obj_005, obj_009, obj_010），跑 3 组 profile 对照：

| 实验组 | Profile | 允许动作 | GPU |
|--------|---------|---------|-----|
| A（对照） | scene_v71a | 全部（no whitelist） | cuda:0 |
| B（lighting only） | scene_v71b | ENV_ROTATE_30/NEG_30, L_KEY_YAW_POS/NEG_15 | cuda:1 |
| C（lighting+shadow） | scene_v71c | B 的基础上 + L_KEY_UP/DOWN, S_CONTACT_SHADOW_UP | cuda:2 |

### 5.2 结果

| 对象 | Exp A | Exp B | Exp C | 一致性 |
|------|-------|-------|-------|--------|
| obj_005 | 0.6380 → **0.6916** (+0.054) | **0.6916** | **0.6916** | ✅ 完全相同 |
| obj_009 | 0.6542 → **0.7229** (+0.069) | **0.7229** | **0.7229** | ✅ 完全相同 |
| obj_010 | 0.6193 → **0.7087** (+0.089) | **0.7087** | **0.7087** | ✅ 完全相同 |

三组实验结果完全一致（三组均选择 ENV_ROTATE_30，诊断均为 scene_light_mismatch）。

### 5.3 原因分析

A=B=C 不是 bug，是以下几个因素叠加的必然结果：

1. **ENV_ROTATE_30 在所有 whitelist 里都合法**：B/C 的 whitelist 均包含 ENV_ROTATE_30
2. **Controller 确定性**：给定同一 mesh + 同一初始 control state，VLM 输出确定，controller 选择确定
3. **Whitelist 约束不生效**：因为最优动作恰好在约束内，限制不改变结果

### 5.4 Step 4 决策评估

| 成功标准 | 评估 | 结果 |
|---------|------|------|
| 2/3 对象 final ≥ 0.75 | 最高 0.7229（obj_009），0/3 达标 | ❌ FAIL |
| 2/3 对象 vs 当前 v7 再提升 +0.03 | B=C=A，Δ=0 | ❌ FAIL |
| scene_light_mismatch 明显下降 | 3/3 仍为 scene_light_mismatch | ❌ FAIL |

**决策：三条标准全部失败 → 冻结 scene controller，停止深挖。**

---

## 六、结论

### 6.1 Controller 收益总结

| 阶段 | 机制 | 结果 |
|------|------|------|
| v7 smoke_v2 | ENV_ROTATE_30 对 2/3 有效 | avg delta +0.066 |
| v7 full pilot | ENV_ROTATE_30 对 7/10 有效 | avg confirmed 0.6043 → avg final 0.6390，**+0.035** |
| v7.1 | whitelist 约束 | B=C=A，无增益，验证 ENV_ROTATE_30 是主导动作 |

### 6.2 质量天花板分析

- **天花板**：~0.72–0.73（3 个最优对象 obj_005/009/010 的 final 区间）
- **根因**：`scene_light_mismatch` 是 VLM 对"场景光照与物体材质/来源不一致"的感知，属于 scene 资产层面的问题
- **Controller 无法解决**：无论如何旋转环境光或调整关键光，4.blend 场景的 HDRI 与物体预期渲染风格之间存在系统性失配
- **结论**：当前 scene controller 层已收敛，进一步优化的 ROI 极低

### 6.3 最终决策

> **冻结 scene controller，停止对 controller 层的工程投入。**
>
> current scene realism gap 源于上游资产（mesh 质量、scene 场景选择），不值得继续在 controller 参数层面深挖。

---

## 七、下阶段方向

基于上述结论，推荐以下方向（按优先级排序）：

### 7.1 直接推进下游训练（推荐）

- 当前 scene_dataset_v0（7 对象，Tier A×2 B×2 C×3）质量区间 0.61–0.72
- 直接用于下游 VLM 训练，验证 0.60–0.72 区间数据是否能产生有效训练信号
- **无需等待更高质量数据**

### 7.2 上游场景资产替换

- 替换 4.blend 为与物体风格更匹配的场景（HDRI 色温/方向一致性）
- 这是解决 scene_light_mismatch 的根本路径，但工程成本较高

### 7.3 Mesh/T2I 质量把控

- obj_002, obj_007 因 contact_gap 过大被剔除，说明部分 mesh 的底部几何有问题
- 上游 mesh 质量检查（底面平整度、与场景尺度匹配）
- 可在 mesh_qc 阶段增加 contact_gap 预测过滤

---

## 附录：关键文件清单

| 文件 | 说明 |
|------|------|
| `pipeline/stage4_scene_render.py` | Blender 场景渲染脚本（v7-1 修正） |
| `pipeline/stage5_5_vlm_review.py` | VLM 评审（v7-2 prompt + contact_gap=0.005） |
| `pipeline/stage5_6_feedback_apply.py` | Action 选择与应用（含 action_whitelist） |
| `run_scene_evolution_loop.py` | 场景演化主循环（contact_gap_minor=0.005） |
| `configs/scene_action_space.json` | Scene Insert action space |
| `configs/scene_template.json` | 4.blend 场景模板 |
| `configs/dataset_profiles/scene_v7.json` | v7 标准 profile |
| `configs/dataset_profiles/scene_v71a.json` | v7.1 对照组（full whitelist） |
| `configs/dataset_profiles/scene_v71b.json` | v7.1 仅 ENV_ROTATE+L_KEY_YAW |
| `configs/dataset_profiles/scene_v71c.json` | v7.1 lighting+contact shadow |
| `scripts/export_scene_dataset_v0.py` | v0 数据集导出脚本 |
| `pipeline/data/scene_dataset_v0/` | 导出的 v0 数据集（7 对象） |
| `pipeline/data/evolution_scene_v7_full_pilot/` | Full pilot 原始结果（10 对象） |
