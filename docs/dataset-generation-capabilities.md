# ARIS Pipeline 数据集生成能力分析

## Context

基于现有 ARIS pipeline 的代码、配置和模板，不考虑具体研究方向，这套工具链能生成什么类型的数据集。以下是从代码逆向推导出的完整能力画像。

---

## 一、Pipeline 能生成的数据集类型

### 1. 多维度变体的合成图像数据集（T2I 路径）

来源：`dataset_synthesis_loop.py` + `t2i_generation.default.json`

pipeline 内置了一个 **prompt 组合引擎**，通过 4 个维度的笛卡尔积自动生成 prompt：

| 维度 | 默认值 | 含义 |
|------|--------|------|
| `subject` | target object, variant object, auxiliary object | 主体物 |
| `scene` | indoor studio, outdoor environment, neutral background | 场景 |
| `lighting` | soft daylight, hard side light, diffuse light | 光照 |
| `style` | photorealistic, documentary, product-shot | 风格 |

- 3 个 prompt 模板 × 4 维度组合 = 最多 **3×3×3×3 = 81 种唯一 prompt**（与烟测中 81 张图吻合）
- `max_unique_prompts` 上限 5000，扩展维度后可大幅增加
- 支持 3 个 T2I provider：`qwen_image_local`（已启用）、`nano_banana`、`gpt_image`
- 每个 provider 可独立生成 preview 8 张 + full 200 张
- 多 provider 并行时，单轮可生成 **200 × 3 = 600 张**

**能生成的数据集特征**：以物体为中心、多场景多光照多风格的合成图像集，适合做图像质量评估、风格一致性评估、场景适配性评估。

### 2. 3D 渲染的多视角数据集（Blender 路径）

来源：`dataset_synthesis_loop.py` + `blender_render.default.json`

- 输入：`.blend` 物体文件夹 + 场景文件
- 输出：图像 + mask + 视频 + metadata JSON
- 规模：preview 8 图 + 2 视频，full **200 图 + 20 视频**
- 支持自定义 Blender 脚本（`command_images` 模板）
- 自动生成对应的 **mask 图**（语义分割/实例分割用）

**能生成的数据集特征**：带 mask 标注的多视角渲染图像+视频，适合做物体检测评估、分割质量评估、3D 一致性评估。

### 3. 双路径融合数据集（Dual 模式）

`--synthesis-mode dual` 同时跑 Blender + T2I，然后 merge：

- Blender 提供精确几何 + mask + 多视角
- T2I 提供风格多样性 + 场景丰富度
- `--data-merge-mode fill-gap`：T2I 补充 Blender 覆盖不到的场景

**能生成的数据集特征**：real + blender + t2i 三源混合数据集，天然带有来源标签，适合做跨域评估、数据源质量对比。

---

## 二、Pipeline 内置的标注能力

### 自动标注（无需人工）

| 标注类型 | 来源 | 说明 |
|----------|------|------|
| 数据来源标签 | pipeline 自动 | `real` / `blender` / `t2i-{provider}` |
| 合成 prompt | T2I 路径 | 每张图对应的生成 prompt |
| Blender metadata | Blender 路径 | 物体名、相机索引、帧数 |
| Mask 图 | Blender 路径 | 像素级分割标注 |
| 双模型评估分数 | `dual_llm_eval` | Claude + GPT-5.4 的 match_score + verdict |
| 缺失数据类型 | 门禁脚本 | `missing_data` 列表（如 rare pose, night scene） |
| QC 过滤结果 | 门禁脚本 | 通过/拒绝 + 原因 |

### 评估 VLM 训练标签（需要训练后产出）

从 `EVAL_MODEL_SPEC.md` 模板，训练后的 VLM 能输出：

```json
{
  "fidelity_score": 0.0,       // 质量/真实感
  "fit_score": 0.0,            // 与任务匹配度
  "controllability_score": 0.0, // prompt/条件一致性
  "reject_reason": "low_fidelity", // 拒绝原因分类
  "final_decision": "reject",  // accept/review/reject
  "rationale": "..."           // 可解释理由
}
```

---

## 三、数据集规模估算

| 配置 | Preview | Full | 说明 |
|------|---------|------|------|
| T2I 单 provider | 8 | 200 | 当前默认 |
| T2I 三 provider | 24 | 600 | 全部启用 |
| Blender 图像 | 8 | 200 | 需要 .blend 资产 |
| Blender 视频 | 2 | 20 | 需要 ffmpeg |
| Dual 模式 | 32 | 800 | T2I(3) + Blender |
| 扩展维度后 | - | **5000+** | 增加 subject/scene 变体 |

门禁要求 ≥1000 样本才能通过。要自然通过门禁，需要：
- 扩展 `variant_dimensions`（增加更多 subject/scene/lighting 变体）
- 或启用多个 provider
- 或多轮合成（heuristic_refine 会自动增加 `full_images_per_provider`）

---

## 四、现成可跑的数据集方向（不需要额外开发）

基于现有代码和配置，以下方向可以直接跑通：

### 方向 A：合成图像质量评估数据集
- 用 T2I 生成多风格图像
- 双模型自动打分（fidelity/fit/controllability）
- 训练 VLM 学习评估合成图像质量
- **所需**：接入至少 1 个真实 T2I provider

### 方向 B：多源数据一致性评估数据集
- Dual 模式生成 blender + t2i 混合数据
- 天然带有来源标签（real/blender/t2i）
- 训练 VLM 判断数据来源 + 评估跨域一致性
- **所需**：接入 T2I provider + Blender 资产

### 方向 C：物体多视角渲染评估数据集
- Blender 路径生成多视角图像 + mask + 视频
- 评估渲染质量、几何一致性、mask 准确性
- **所需**：准备 .blend 物体和场景文件

### 方向 D：Prompt-图像对齐评估数据集
- T2I prompt 组合引擎生成 prompt-image pairs
- 双模型评估 prompt 与图像的匹配度
- 训练 VLM 做 prompt faithfulness 评估
- **所需**：接入至少 1 个真实 T2I provider

---

## 五、启动任何方向的最小行动

无论选哪个方向，都需要先完成这 3 步：

1. **写 `FINAL_PROPOSAL.md`**：运行 `/idea-discovery "选定的方向描述"`
2. **接入至少 1 个真实 API**：
   - 最简路径：配置 `qwen_image_local` 的 command（本地 qwen 模型）
   - 或配置 `gpt_image` / `nano_banana` 的 API
3. **扩展 variant_dimensions**：把默认的 3×3×3×3=81 扩展到 ≥1000 以通过门禁

---

## 六、验证方式

分析结论可通过以下方式验证：
- 检查 `t2i_generation.default.json` 的 variant_dimensions 笛卡尔积 = 81（与烟测吻合）
- 检查 `synthesis_manifest.json` 确认 81 张图的生成记录
- 检查 `blender_render.default.json` 确认 200 图 + 20 视频的默认配置
- 检查 `EVAL_MODEL_SPEC.md` 确认输出 schema 的 5 个字段

---

## 生成时间

2026-03-20
