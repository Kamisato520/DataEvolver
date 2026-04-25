# VLM Review 预设配置解析

> 文件：`pipeline/stage5_5_vlm_review.py`
> 描述：在渲染图像送给 VLM（Qwen3.5-35B-A3B）进行 review、AI agent 正式接手之前，系统所完成的全部预设配置逻辑。

---

## 概览：配置的四个层次

```
review_object()
  │
  ├─ 【第一层】全局常量（模型路径 / 视角 / 权重）
  ├─ 【第二层】GPU 显存分配策略（_resolve_vlm_load_kwargs）
  ├─ 【第三层】VLM 行为模式预决策（freeform / thinking 开关）
  │
  └─ 对每个 view (az, el)：
       │
       ├─ CV 传统指标预计算（exposure / sharpness / mask）
       │
       └─ run_vlm_review()
            │
            ├─ _resolve_vlm_load_kwargs()   ← GPU 分配配置
            ├─ load_vlm()                   ← 模型加载（带全局缓存）
            ├─ 组装图片内容列表（content）
            ├─ 【第四层】_build_prompt()    ← Prompt 预组装 + 约束注入
            │
            └─ model.generate()            ← VLM 正式接手
```

---

## 第一层：全局常量配置

**位置**：`stage5_5_vlm_review.py` 顶部，约 L44–91

### 模型与路径

```python
VLM_MODEL_PATH = "/data/wuwenzhuo/Qwen3.5-35B-A3B"
VLM_MODEL_NAME = "Qwen3.5-35B-A3B"

# 普通对象 / 场景插入分别对应不同的 action space 配置文件
ACTION_SPACE_PATH       = os.path.join(DATA_BUILD_ROOT, "configs", "action_space.json")
SCENE_ACTION_SPACE_PATH = os.path.join(DATA_BUILD_ROOT, "configs", "scene_action_space.json")
```

### 代表性视角（Rep Views）

```python
# 普通对象：4 个正交水平视角
REP_VIEWS = [(0, 0), (90, 0), (180, 0), (270, 0)]

# 场景插入：加入仰角变化，更贴近真实拍摄角度
SCENE_REP_VIEWS = [(0, 0), (90, 0), (180, 15), (45, 30)]
```

### 混合评分权重

VLM 语义分与 CV 传统指标分按以下权重融合：

| 权重组 | 字段 | 权重 |
|--------|------|------|
| VLM 分 | `lighting` | 0.25 |
| VLM 分 | `object_integrity` | 0.30 |
| VLM 分 | `composition` | 0.20 |
| VLM 分 | `render_quality_semantic` | 0.10 |
| VLM 分 | `overall` | 0.15 |
| CV 分（有 mask） | `mask_score` | 0.30 |
| CV 分（有 mask） | `exposure_score` | 0.25 |
| CV 分（有 mask） | `sharpness_score` | 0.20 |
| CV 分（有 mask） | `framing_score` | 0.25 |
| CV 分（无 mask） | `exposure_score` | 0.556 |
| CV 分（无 mask） | `sharpness_score` | 0.444 |

> **设计意图**：全局常量层决定"评什么、用哪个模型、怎么算分"，是所有后续步骤的基准。修改权重会直接影响最终 `hybrid_score` 的计算结果。

---

## 第二层：GPU 显存分配策略

**函数**：`_resolve_vlm_load_kwargs(device: str) -> dict`（L439–472）

在模型加载之前，该函数通过读取环境变量决定显存分配方案：

```python
def _resolve_vlm_load_kwargs(device: str) -> dict:
    force_single   = os.environ.get("VLM_FORCE_SINGLE_GPU", "0")   # 强制单卡
    allow_balanced = os.environ.get("VLM_ALLOW_BALANCED",   "0")   # 允许跨卡均衡
    num_visible    = torch.cuda.device_count()                      # 可见 GPU 数量
    max_mem_gib    = os.environ.get("VLM_MAX_MEMORY_GIB",   "75")  # 每卡显存上限
```

### 决策逻辑

```
环境变量 VLM_FORCE_SINGLE_GPU=1
    → device_map = <指定设备>，单卡加载

环境变量 CUDA_VISIBLE_DEVICES 只有 1 张卡
    → device_map = <指定设备>，单卡加载

num_visible > 1 且未强制单卡
    → device_map = "balanced"
    → max_memory = {0: "75GiB", 1: "75GiB", ...}  （可由 VLM_MAX_MEMORY_GIB 调整）
    → 跨卡均衡分配，适用于 8×H100 环境

其余情况
    → 回退到 device_map = <指定设备>
```

所有情况均附带 `low_cpu_mem_usage: True`，避免在 CPU 内存中冗余驻留权重。

> **设计意图**：Qwen3.5-35B-A3B 是 MoE 架构，单张 H100（80 GiB）放不下全量激活，因此需要跨卡均衡。该函数将硬件决策与业务逻辑完全解耦，只需设置环境变量即可切换部署模式。

---

## 第三层：VLM 行为模式预决策

**位置**：`review_object()` 函数内，L1666–1685

在遍历各视角、调用 VLM 之前，系统预先决定两个核心行为开关，并完成上下文资源的解析：

### 两个核心开关

```python
# 开关 1：VLM 输出模式
use_freeform_vlm_feedback = bool(
    (profile_cfg or {}).get("use_freeform_vlm_feedback",
    review_mode == "scene_insert")   # scene_insert 模式默认开启
)

# 开关 2：VLM 内部推理模式
enable_vlm_thinking = bool(
    (profile_cfg or {}).get("enable_vlm_thinking",
    review_mode == "scene_insert")   # scene_insert 模式默认开启
)
```

| 开关 | 来源优先级 | 默认值（scene_insert） | 默认值（studio） |
|------|-----------|----------------------|----------------|
| `use_freeform_vlm_feedback` | `profile_cfg` > `review_mode` | `True` | `False` |
| `enable_vlm_thinking` | `profile_cfg` > `review_mode` | `True` | `False` |

**`use_freeform_vlm_feedback = True` 的推理路径**：
```
VLM 先自由输出文字分析（freeform）
  → _coerce_freeform_to_json() 尝试将自然语言转为 JSON
  → 若失败，_heuristic_review_from_freeform() 启发式提取关键字段
  → parse_mode 记录为 "repaired_from_freeform" 或 "heuristic_from_freeform"
```

**`use_freeform_vlm_feedback = False` 的推理路径**：
```
VLM 直接输出 JSON（prompt 中预填 "{" 强制以大括号开头）
  → _extract_json() 直接解析
  → 若失败，_coerce_freeform_to_json() 作为 fallback
  → parse_mode 记录为 "direct_json" 或 "repaired_from_json_failure"
```

### 上下文资源解析

同步完成以下资源路径的解析（若未由调用方显式传入）：

```python
# 原始参考图：用于对比 structure / color 一致性
reference_image_path = _resolve_reference_image(obj_id, profile_cfg, renders_dir)

# 伪参考图：场景融合的粗目标样例，非严格 ground truth
pseudo_reference_path = _resolve_pseudo_reference_image(obj_id, profile_cfg, renders_dir)

# 根据 review_mode 自动选择 action space 配置文件
resolved_action_space = action_space_path or (
    SCENE_ACTION_SPACE_PATH if review_mode == "scene_insert" else ACTION_SPACE_PATH
)
```

> **设计意图**：`profile_cfg` 是运行时传入的个性化配置字典，允许在不修改代码的情况下为不同资产或项目切换行为模式。`review_mode` 作为兜底默认值，保证 scene_insert 场景默认启用更强的推理能力（freeform + thinking），studio 场景则优先速度。

---

## 第四层：Prompt 预组装与约束注入

**函数**：`_build_common_prompt()`（L527–702），由 `_build_prompt()` 或 `_build_freeform_prompt()` 调用

这是 VLM 接手前最关键的一层——决定 VLM "看到什么、能说什么、必须怎么说"。

### 4.1 System Message：角色定位

```python
system_msg = (
    "You are a practical render reviewer helping improve iterative 3D scene insertion results. "
    "Your primary goal is to judge whether the current image looks convincing and usable "
    "to a human viewer. "
    "Use the numeric scores only as coarse bookkeeping, not as the main objective. "
    "Return ONLY a valid JSON object, no other text, no markdown fences. "
    "The first character of your reply must be '{' and the last character must be '}'."
)
```

**关键设计**：
- **人类视觉优先**：不追求指标最优，追求"人类观看者觉得可信"
- **JSON 强约束**：从角色定义层面就要求纯 JSON 输出，与 prompt 末尾的格式模板双重锁定

### 4.2 图片上下文描述：动态拼接

```python
image_desc_parts = [f"Image 1 is the CURRENT render (view az={az}, el={el})."]

if has_reference:      # 有原始参考图
    image_desc_parts.append(f"Image {next_idx} is the ORIGINAL REFERENCE image "
                             "(object identity, texture, color reference).")
if has_pseudo_reference:  # 有伪参考图
    image_desc_parts.append(f"Image {next_idx} is the PSEUDO-REFERENCE image "
                             "(rough target scene integration example, not strict ground truth).")
if has_prev:           # 有上一轮渲染
    image_desc_parts.append(f"Image {next_idx} is the PREVIOUS render (round {round_idx-1}).")
```

VLM 收到的图片序列顺序固定为：**当前渲染 → 原始参考 → 伪参考 → 上一轮渲染**，并用文字明确标注每张图片的身份，防止混淆。

### 4.3 打分标准：按 review_mode 分支

**studio 模式**（标准 3D 渲染质量评估）：

```
lighting:               整体照明质量，无过强阴影，对比度合适
object_integrity:       对象完整，无穿模，几何正确
composition:            居中，适当比例，无截断
render_quality_semantic: 无伪影、模糊、贴图问题
overall:                数据集可用性整体评分
```

**scene_insert 模式**（场景融合质量评估，额外优先级规则）：

```
Decision priority:
  1. 以最终视觉效果判断：人类审阅者会保留这张图吗？
  2. 优先整体真实感，而非指标精度
  3. 小瑕疵不过度惩罚，只要整体效果够好
  4. 明显失败才给低分
  5. 如果当前帧比上一帧明显更好，pairwise_vs_prev 应如实反映
```

### 4.4 Action Space 约束注入

```python
allowed_actions_str = ", ".join(
    _allowed_actions_for_group(active_group, action_space_path)
)
# 示例输出："NO_OP, raise_key_light, lower_key_light, add_rim_light, ..."
```

- VLM 的 `suggested_actions` 字段被限制为**只能从当前 active_group 的可用 action 列表中选择**
- 列表从 `action_space.json` / `scene_action_space.json` 动态读取
- 防止 VLM 自由发挥推荐不存在或属于其他 group 的操作

### 4.5 输出 JSON 模板：枚举锁定

Prompt 末尾直接嵌入完整的 JSON 模板，所有枚举值都预先声明：

```json
{
  "schema_version": "vlm_review_v1",
  "sample_id": "<obj_001_az000_el+00>",
  "round_idx": 0,
  "vlm_route": "<pass|needs_fix|reject>",
  "scores": {
    "lighting": "<1-5>",
    "object_integrity": "<1-5>",
    "composition": "<1-5>",
    "render_quality_semantic": "<1-5>",
    "overall": "<1-5>"
  },
  "confidence": {"lighting": "<low|medium|high>", ...},
  "issue_tags": ["<最多 3 个，来自白名单>"],
  "suggested_actions": ["<最多 2 个，来自 active_group>"],
  "lighting_diagnosis": "<flat_no_rim|flat_low_contrast|underexposed_global|...>",
  "structure_consistency": "<good|minor_mismatch|major_mismatch>",
  "color_consistency": "<good|minor_shift|major_shift>",
  "physics_consistency": "<good|minor_issue|major_issue>",
  "asset_viability": "<continue|abandon|unclear>",
  "abandon_reason": "<short reason or null>",
  "abandon_confidence": "<low|medium|high|null>",
  "pairwise_vs_prev": {
    "available": true,
    "winner": "<current|previous|tie|none>",
    "lighting": "<better|same|worse|na>",
    ...
  }
}
```

> **设计意图**：将所有允许的枚举值直接写进 prompt，配合 `text_input = text_input + "{"` 的前缀注入技巧（强制 VLM 以 `{` 开头输出），最大程度压缩输出空间，提高 JSON 解析成功率，降低后处理修复成本。

---

## 完整流程时序图

```
调用方传入: obj_id, renders_dir, round_idx, active_group,
            review_mode, profile_cfg, device
       │
       ▼
review_object()
  ├─ [配置] 解析参考图路径（reference / pseudo_reference）
  ├─ [配置] 选择 action_space 文件（studio vs scene_insert）
  ├─ [配置] 预决策 use_freeform / enable_thinking
  │
  └─ for (az, el) in views:
       │
       ├─ [预处理] compute_cv_metrics()
       │    ├─ compute_mask_score()
       │    ├─ compute_exposure_score()
       │    ├─ compute_sharpness_score()
       │    └─ compute_framing_score()
       │
       └─ run_vlm_review()
            │
            ├─ [硬件] _resolve_vlm_load_kwargs()
            ├─ [加载] load_vlm()（全局缓存，只加载一次）
            ├─ [上下文] 组装 content 列表（图片顺序固定）
            │
            ├─ if use_freeform_first:
            │    ├─ _build_freeform_prompt()
            │    ├─ model.generate()         ← VLM 接手（freeform）
            │    ├─ _coerce_freeform_to_json()
            │    └─ _heuristic_review_from_freeform()（fallback）
            │
            └─ else:
                 ├─ _build_prompt()          ← 第四层 Prompt 预组装
                 ├─ text_input += "{"        ← 前缀注入
                 ├─ model.generate()         ← VLM 接手（JSON 直出）
                 ├─ _extract_json()
                 └─ _coerce_freeform_to_json()（fallback）
                      │
                      └─ _validate_review()  ← 输出验证 & 字段修正
```

---

## 关键设计原则总结

| 原则 | 体现 |
|------|------|
| **硬件与业务解耦** | `_resolve_vlm_load_kwargs` 通过环境变量控制 GPU 分配，业务代码无需关心硬件拓扑 |
| **模式可切换** | `use_freeform_vlm_feedback` / `enable_thinking` 由 `profile_cfg` 或 `review_mode` 决定，零代码改动即可切换 |
| **输出空间压缩** | 枚举白名单 + JSON 模板 + `"{"` 前缀注入，三重约束限制 VLM 输出，提升解析成功率 |
| **多级 fallback** | direct_json → repaired_from_freeform → heuristic → neutral_fallback，任何阶段失败都不中断流程 |
| **人类视觉优先** | system_msg 明确要求以"人类观看者觉得可信"为首要标准，数值分只是辅助记录 |
| **可追溯性** | `vlm_dialogue` trace 记录每次尝试的 system/user prompt 和原始输出，写入独立 JSON 文件 |