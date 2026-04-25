# VLM 反馈维度与渲染参数自动调整机制解析（修订版）

> 涉及文件：
> - `deliverables/vlm_preset_rule_bundle_20260410/configs/vlm_review_schema.json` — VLM 输出 schema
> - `deliverables/vlm_preset_rule_bundle_20260410/pipeline/stage5_5_vlm_review.py` — VLM review 阶段
> - `deliverables/vlm_preset_rule_bundle_20260410/configs/scene_action_space.json` — 动作空间配置
> - `deliverables/vlm_preset_rule_bundle_20260410/scripts/run_scene_agent_monitor.py` — 主控 monitor
> - `deliverables/vlm_preset_rule_bundle_20260410/scripts/run_scene_agent_step.py` — 单轮执行器

---

## 前言：之前分析的两个根本性错误

之前的分析把这套系统描述为"纯规则查表系统"，这是错误的。正确理解如下：

| 错误理解 | 正确理解 |
|---------|---------|
| monitor 只读结构化 JSON 字段，做简单查表 | monitor **优先读 VLM 的自由文本**（`assistant_text`），结构化字段是辅助 |
| `decide_actions()` 是 diagnosis → action 的一对一映射 | `decide_actions()` 是一个富启发式解析器，从自然语言中提取几十种语义信号 |
| 整个流程是确定性规则系统 | 流程是"VLM 自由评价 + 脚本语义解析 + 规则约束"的混合系统 |

---

## 一、VLM 从哪些维度进行反馈？这些维度是人为定义的吗？

### 1.1 结论

> 反馈维度的**字段结构和枚举值**由人完全预定义（schema 锁死）。
> 但 VLM 同时会输出**自由文本**（freeform assistant_text），这才是后续 monitor 的主要决策依据。
> 也就是说：**结构是人定的，内容中最重要的部分是 VLM 自由生成的**。

---

### 1.2 VLM 的双轨输出

VLM 在这套系统里实际上产出两类输出：

```
VLM 输出
  ├── Track 1: 结构化 JSON（schema 严格约束）
  │     → 每个字段、每个枚举值都是人预定义的
  │     → 供脚本精确解析
  │
  └── Track 2: 自由文本（freeform assistant_text）
        → VLM 自行组织的自然语言评价
        → monitor 的 decide_actions() 以此为主要输入
```

当 `use_freeform_vlm_feedback=True`（scene_insert 模式默认开启）时，VLM 先写自由文字评价，
再由 `_coerce_freeform_to_json()` 将其转换为结构化 JSON。自由文本同步保存进 trace 文件。

---

### 1.3 结构化 JSON 的所有字段（schema 预定义）

这些字段完全由 `configs/vlm_review_schema.json` 定义，VLM 不能增减字段：

#### 评分类（5 个主维度，1–5 分）

| 字段 | 含义 | 评价重点 |
|------|------|---------|
| `lighting` | 光照质量 | 亮度、阴影、与场景光源匹配度、整体照明自然度 |
| `object_integrity` | 几何完整性 | 物体是否完整、有无畸形/断裂/低模崩坏/材质身份错误 |
| `composition` | 构图 | 大小、位置、接地感、与背景关系、是否有 pasted-on 感 |
| `render_quality_semantic` | 渲染语义质量 | 渲染出来的东西是否像它应该是的那个物体（画质 + 语义身份） |
| `overall` | 总体印象分 | 人类审阅者的整体直觉评价 |

这 5 个维度的混合权重在代码中硬编码：

```python
VLM_W = {
    "lighting": 0.25,
    "object_integrity": 0.30,
    "composition": 0.20,
    "render_quality_semantic": 0.10,
    "overall": 0.15
}
```

#### 诊断类（4 个枚举字段）

| 字段 | 枚举值 | 作用 |
|------|--------|------|
| `vlm_route` | `pass / needs_fix / reject` | 整体处置信号 |
| `lighting_diagnosis` | `flat_no_rim / flat_low_contrast / underexposed_global / underexposed_shadow / harsh_shadow_key / scene_light_mismatch / shadow_missing / good` | 光照问题子类型定位 |
| `structure_consistency` | `good / minor_mismatch / major_mismatch` | 3D 结构与参考图一致性 |
| `color_consistency` | `good / minor_shift / major_shift` | 颜色/材质与参考图偏差 |
| `physics_consistency` | `good / minor_issue / major_issue` | 物理合理性（悬空/穿地） |
| `asset_viability` | `continue / abandon / unclear` | 资产是否仍有优化价值 |

#### 可操作信号类（2 个列表字段）

| 字段 | 约束 | 说明 |
|------|------|------|
| `issue_tags` | 最多 3 个，来自 36 个预定义标签 | 具体视觉问题标签 |
| `suggested_actions` | 最多 2 个，来自 52 个预定义动作名 | VLM 自己建议的下一步动作 |

`suggested_actions` 的枚举范围（52 个，schema 中完整定义）比 `scene_action_space.json` 实际执行的动作集更大，
说明 schema 预留了比当前动作空间更宽的建议空间。

#### 比较类（1 个对象字段）

| 字段 | 说明 |
|------|------|
| `pairwise_vs_prev` | 与上一轮逐维度对比：`better / same / worse / na`，以及总体 `winner` 判断 |
| `confidence` | 每个评分维度的置信度：`low / medium / high` |

---

### 1.4 为什么不让 VLM 自由生成维度？

结构化字段被 schema 锁死，是为了保证**可解析性**——每条诊断字段都对应 action_space 中的一套预设动作映射。

但系统并不完全依赖这些结构化字段做决策。自由文本（Track 2）是真正驱动 monitor 的主要信号。
这是"人定结构、VLM 自由表达、脚本语义解析"三者分工的设计。

---

## 二、Codex 正式下场前，脚本如何根据 VLM 反馈自动调整渲染参数？

### 2.1 整体链路

```
VLM review（每个视角）
  ├── 结构化 JSON → 写入 per-view review 文件
  └── freeform assistant_text → 写入 trace 文件

多视角聚合 → agg.json（含聚合分数 + 最常见 issue_tags + lighting_diagnosis 等）

run_scene_agent_monitor.py（主控循环）
  │
  ├── extract_trace_text()         ← 读取 freeform assistant_text（主要输入）
  ├── build_decision_payload()
  │     ├── detect_verdict()       ← 判断是否已达到 "keep" 结论
  │     ├── detect_asset_viability() ← 判断是否应放弃资产
  │     └── decide_actions()       ← 核心：从 freeform 文本 + 结构化字段中选动作
  │
  ├── evaluate_deprecation_for_pair()  ← 多维度弃用判断
  │
  └── run_one_round()              ← 执行下一轮渲染
        └── run_scene_agent_step.py ← 单轮执行器：应用动作 → Blender 渲染 → VLM review
```

---

### 2.2 `decide_actions()`：核心决策函数

**这是整个系统最关键的函数，也是之前分析最大的误解所在。**

它**不是**一个简单的 `diagnosis → action` 查表，而是一个富语义解析器：

```python
def decide_actions(agg: dict, trace_text: str, round_idx: int):
    text = (trace_text or "").lower()          # 自由文本（主要）
    issues = {str(v).lower() for v in (agg.get("issue_tags") or [])}  # 结构化标签（辅助）
    suggestions = _suggested_actions(agg)      # VLM 建议动作（补充）
    physics = agg.get("programmatic_physics")  # 程序化物理检测结果
    lighting = str(agg.get("lighting_diagnosis") or "").lower()
    color_consistency = str(agg.get("color_consistency") or "").lower()
```

#### 语义信号提取（从自由文本中解析的典型信号）

函数首先从 `trace_text` 中提取约 20 类语义信号：

| 信号变量 | 触发关键词（部分示例） | 对应问题 |
|---------|---------------------|---------|
| `too_small` | "too small", "miniature", "toy", "undersized" | 物体偏小 |
| `too_large` | "too large", "oversized", "giant", "huge" | 物体偏大 |
| `too_dark` | "too dark", "too black", "pitch black", "much darker" | 整体过暗 |
| `detail_obscured` | "silhouetted", "hard to see details", "details are lost" | 暗部细节丢失 |
| `too_bright` | "too bright", "too light", "washed out" | 整体过亮 |
| `pink_or_purple` | "pinkish", "pink/purple", "magenta", "violet" | 色偏粉紫 |
| `too_warm` | "too orange", "too reddish", "too red", "too yellow" | 色偏暖 |
| `too_cool` | "too blue", "too cool", "bluish", "purple cast" | 色偏冷 |
| `explicit_plastic` | "plastic", "cgi-like", "sticker", "pasted on", "too smooth" | 材质塑料感 |
| `explicit_matte` | "too matte", "lacks highlights", "flat white stripes" | 材质过哑 |
| `wants_shinier` | "shinier", "more reflective", "glossier", "specular" | 需要更多高光 |
| `strange_texture` | "cracked", "veiny", "solid wheels", "strange texture" | 贴图异常 |
| `ground_intersection` | "intersecting the ground", "sinking into", "cuts into the road" | 穿地 |
| `explicit_floating` | "floating", "hovering", "not contacting" | 悬空 |
| `flat_lighting` | "flat lighting", "low contrast" | 光照过平 |
| `strong_god_rays` | "god rays", "sunbeams", "volumetric light", "too foggy" | 体积光过强 |
| `weak_separation` | "doesn't stand out", "blends into the scene" | 主体分离弱 |

这些信号**同时结合** `issue_tags` 和 `lighting_diagnosis` 做交叉验证：

```python
too_dark = "underexposed" in issues or detail_obscured or _contains_any(text, ["too dark", ...])
```

#### 冲突消解：信号互斥处理

当同一物理量的两个方向信号同时触发时，有明确的优先级规则：

```python
if too_small and too_large:
    if "object_too_small" in issues and "object_too_large" not in issues:
        too_large = False    # 结构化标签优先仲裁
    else:
        too_small = False; too_large = False  # 两个都不确定则都清除

if too_dark and too_bright:
    # 同样逻辑，issue_tags 优先仲裁
```

#### 动作黑名单（blocked set）：防止矛盾动作被执行

在选动作之前，先根据语义信号预置黑名单：

```python
if ground_intersection:
    blocked.add("O_LOWER_SMALL")      # 已经穿地，不能再往下压
if explicit_floating:
    blocked.add("O_LIFT_SMALL")       # 已经悬空，不能再往上抬
if wants_shinier or too_matte:
    blocked.add("M_ROUGHNESS_UP")     # 想要更亮高光，不能加粗糙度
if flat_lighting and not needs_more_light:
    blocked.add("ENV_STRENGTH_UP")    # 光照过平但不是太暗，不该加环境光
if strong_god_rays:
    blocked.add("ENV_STRENGTH_UP")
    blocked.add("L_KEY_UP")           # 体积光过强，不能继续加光
```

#### 动作选择顺序

按优先级依次加入动作列表（最终截取前 4 个）：

```
Priority 1: 物体尺度问题（too_small → SCALE_UP_10，too_large → SCALE_DOWN_10）
Priority 2: 接地/悬浮问题（weak_ground → CONTACT_SHADOW_UP；floating → LOWER_SMALL）
Priority 3: VLM 自身建议动作（suggestions，来自结构化字段 suggested_actions）
Priority 4: 光照亮度问题（needs_more_light → ENV_STRENGTH_UP, M_VALUE_UP, L_KEY_UP）
Priority 5: 色彩偏差问题（too_warm → HUE_WARM_NEG；wants_warmer → HUE_WARM_POS）
Priority 6: 材质质感问题（explicit_plastic → ROUGHNESS_UP；wants_shinier → SHEEN_UP）
Priority 7: 主体分离/光照平（flat_lighting → L_KEY_UP）
Priority 8: 体积光过强（strong_god_rays → ENV_STRENGTH_DOWN, L_KEY_DOWN）
Fallback:   无明确信号时 → CONTACT_SHADOW_UP + L_KEY_UP（默认安全动作）
```

---

### 2.3 `_prune_conflicting_actions()`：最终冲突裁剪

选完动作后，再做一次互斥裁剪（防止列表内部矛盾）：

```python
conflicts = {
    "O_SCALE_UP_10":        {"O_SCALE_DOWN_10"},
    "O_LIFT_SMALL":         {"O_LOWER_SMALL"},
    "M_VALUE_UP":           {"M_VALUE_DOWN", "M_VALUE_DOWN_STRONG"},
    "M_HUE_WARM_POS":       {"M_HUE_WARM_NEG"},
    "M_ROUGHNESS_UP":       {"M_ROUGHNESS_DOWN"},
    ...
}
```

先进入列表的动作优先保留，后进入的如果与已有动作冲突则丢弃。

---

### 2.4 `build_decision_payload()`：决策包

每轮的完整决策结果会落盘为 `decisions/round{N}_decision.json`：

```json
{
  "detected_verdict": "needs_fix",
  "asset_viability": "continue",
  "hybrid_score": 0.612,
  "issue_tags": ["floating_visible", "color_shift"],
  "lighting_diagnosis": "shadow_missing",
  "programmatic_physics": {"is_floating": true, "contact_gap": 0.015},
  "reviewer_excerpt": "The object appears to be hovering slightly above the ground...",
  "chosen_actions": ["O_LOWER_SMALL", "S_CONTACT_SHADOW_UP", "M_VALUE_DOWN"],
  "decision_reasons": [
    "程序物理显示有接触间隙，轻微下压",
    "VLM 反复指出接地差/阴影弱，先加强接触阴影",
    "VLM 说过亮/过浅，先压低材质亮度"
  ]
}
```

---

### 2.5 循环的退出条件

monitor 的 `process_pair()` 每轮检查以下退出条件（优先级从高到低）：

| 退出条件 | 触发机制 | 最终状态 |
|---------|---------|---------|
| `detected_verdict == "keep"` | VLM 自由文本中出现明确的 keep 信号 | `status: kept`，成功结束 |
| `deprecation_check.should_deprecate` | 资产被判定为无法修复（材质白模/网格损坏等） | `status: deprecated_enqueued`，加入待替换队列 |
| `round_idx >= HARD_MAX_ROUNDS (15)` | 达到硬性轮次上限 | `status: max_rounds_reached`，退出 |
| `render_failed` | Blender 渲染报错 | `status: deprecated_enqueued` |
| 网格文件缺失/为空 | 启动时检查 mesh 文件 | `status: deprecated_enqueued` |

`HARD_MAX_ROUNDS = 15` 是系统级硬限制，保证不会无限迭代追求边际改善。

---

### 2.6 动作执行：`apply_action()` 的数值计算

选定动作后，由 `stage5_6_feedback_apply.py` 的 `apply_action()` 执行参数更新：

```python
# mul 类型（灯光强度、缩放比例等）
effective_delta = (raw_delta - 1.0) * step_scale + 1.0
new_val = current * effective_delta

# add 类型（角度、偏移量等）
effective_delta = raw_delta * step_scale
new_val = current + effective_delta

# 钳位到预设范围
new_val = max(lo, min(hi, new_val))
```

步长随轮次衰减（score-adaptive）：

| 轮次 | 基础步长 | 低分时（hybrid_score < 0.65）步长 |
|------|---------|--------------------------------|
| Round 0 | 100% | 120% |
| Round 1 | 70% | 84% |
| Round 2 | 50% | 60% |
| Round 3+ | 40% | 48% |

---

### 2.7 可调参数的完整空间（`scene_action_space.json`）

所有自动调整都限制在以下离散动作空间内：

#### lighting 组

| 动作 | 参数 | 幅度 | 范围 |
|------|------|------|------|
| `L_KEY_UP / DOWN` | `key_scale` | ×1.2 / ×0.8 | [0.5, 2.0] |
| `L_KEY_YAW_POS/NEG_15` | `key_yaw_deg` | ±15° | [-90°, 90°] |

#### object 组

| 动作 | 参数 | 幅度 | 范围 |
|------|------|------|------|
| `O_LIFT/LOWER_SMALL` | `offset_z` | ±0.02 | [-0.1, 0.1] |
| `O_ROTATE_Z_POS/NEG_15` | `yaw_deg` | ±15° | [-45°, 45°] |
| `O_SCALE_UP/DOWN_10` | `scale` | ×1.1 / ×0.9 | [0.7, 1.4] |

#### scene 组

| 动作 | 参数 | 幅度 | 范围 |
|------|------|------|------|
| `ENV_ROTATE_30 / NEG_30` | `hdri_yaw_deg` | ±30° | [-180°, 180°] |
| `ENV_STRENGTH_UP / DOWN` | `env_strength_scale` | ×1.2 / ×0.8 | [0.5, 2.0] |
| `S_CONTACT_SHADOW_UP` | `contact_shadow_strength` | ×1.2 | [0.5, 2.0] |

#### material 组

| 动作 | 参数 | 幅度 | 范围 |
|------|------|------|------|
| `M_SATURATION_DOWN` | `saturation_scale` | ×0.9 | [0.5, 1.5] |
| `M_VALUE_UP / DOWN / DOWN_STRONG` | `value_scale` | ×1.1 / ×0.9 / ×0.8 | [0.4, 1.5] |
| `M_HUE_WARM_NEG / POS` | `hue_offset` | ±0.03 | [-0.15, 0.15] |
| `M_ROUGHNESS_UP / DOWN` | `roughness_add` | ±0.08 | [-0.3, 0.6] |
| `M_SHEEN_UP` | `specular_add` | +0.08 | [0.0, 0.5] |

调整结果写入 `control_state.json`，由 Blender 脚本读取后执行下一轮渲染。

---

## 三、为什么之后还需要 AI agent 亲自接手？

README 中明确说明了预设规则阶段的局限：

> 有些 case 的问题不是简单亮一点、暗一点、缩一点就能解决。
> 对"材质身份错误 / 白模无颜色 / 资产级问题"不够强。

当 monitor 以 `max_rounds_reached` 退出（或遇到预设规则无法处理的难例）后，
AI agent（Codex/Claude）亲自介入，能做预设规则做不到的事：

| 能力维度 | 预设规则阶段 | AI agent 接手后 |
|---------|------------|----------------|
| 输入 | 结构化 JSON + 自由文本关键词匹配 | **直接阅读完整 VLM 自由文本，不做关键词过滤** |
| 动作空间 | 只能用 action_space 里的离散动作 | **可以直接修改 control_state 任意字段** |
| 代码修改 | 不能修改 | **可以直接改渲染逻辑或材质适配代码** |
| 推理深度 | 规则匹配，无推理 | **理解问题根因，制定多步策略** |

---

## 四、核心总结

### 问题 1：VLM 反馈维度是人定的还是 AI 生成的？

**结构是人定的，内容中最有价值的部分是 VLM 自由生成的。**

- `vlm_review_schema.json` 锁定了所有字段名和枚举值，VLM 不能增减维度
- 但 VLM 同时输出 freeform assistant_text，这才是 monitor 决策的主要依据
- 设计分工：人负责定义"要评什么"，VLM 负责"自由表达发现了什么"

### 问题 2：Codex 下场前，脚本如何把 VLM 反馈转成渲染参数调整？

**核心链路是"VLM 自由文本 → 富语义解析 → 动作选择 → 数值计算 → control_state.json"。**

- `decide_actions()` 优先解析 VLM 自由文本中的几十类语义关键词，而不是简单查表
- 结构化字段（issue_tags、suggested_actions 等）作为辅助信号参与决策
- 动作黑名单和 `_prune_conflicting_actions()` 保证不执行矛盾动作
- 最多迭代 15 轮（`HARD_MAX_ROUNDS`），之后由 AI agent 亲自接手难例