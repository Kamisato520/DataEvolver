# GPT "成品"评价：REVIEW_OF_GPT_FINAL

**审阅对象**：`gpt's report/scene_aware_workflow_report_en.md`（228 行）+ 中文版 + `references_verification.md`
**对照金标准**：`NARRATIVE_ARC.md`（172 行）+ `REVIEW_OF_DRAFTS.md`（397 行）+ `CLAUDE.md` 中 R1 实验数字
**审阅日期**：2026-04-24
**审阅者**：Claude Sonnet（基于 NARRATIVE_ARC 九条硬约束全量比对）

---

## 执行摘要

这版"成品"相比之前的草稿有**实质性改进**，在叙事主线对齐、数字诚实度、格式结构上都达到了可投稿的基线水平。但仍存在若干可识别的缺陷，最致命的两条是：**①叙事顺序在 Section 1 就把 rotation/azimuth 术语深度嵌入，违反约束 #1**；**②参考文献 [10] 引用了一个极大概率是幻觉的 arXiv ID（2508.02324 的"Qwen-Image Technical Report"在 2025 年 8 月才出现，与 CLAUDE.md 描述的模型时间线不符，且 Qwen-Image-Edit-2511 实际是 2024 年发布）**。总体而言判定为 **major-revision**，不能直接发 arxiv。

---

## 一、Thesis 对齐度

### 1.1 正向发现

这版成品在 Thesis 对齐度上有显著提升，明显优于 REVIEW_OF_DRAFTS.md 中批评的旧草稿。

**标题（第 1 行）**  
`From Manual Blender Tuning to Self-Evolving Data Construction: A VLM- and Agent-Guided Workflow-as-Skill Framework for Object-Level Image Editing`

标题完整保留了 NARRATIVE_ARC 英文 tagline 的核心词汇（"VLM- and Agent-Guided"、"Workflow-as-Skill"），后缀是"Object-Level Image Editing"而非"Object Rotation Editing"，实现了通用化——这是旧草稿被批评的第一个问题，这版已经修复。

**Abstract（第 3–5 行）**  
Abstract 第一句就写"closed-loop process of observation, diagnosis, parameter adjustment, regeneration, and validation"，直接呼应了 NARRATIVE_ARC 长版 Thesis 的逐字表述。`inspect` verdict 在 Abstract 末句被明确定位为"diagnostic outcome"而非失败，完全符合约束 #4。

**Section 3（第 37–47 行）**  
`Skill = (Stages, Controller, Review, Verdict)` 的形式化定义在这版终于出现，正是 REVIEW_OF_DRAFTS.md 动作 3 所要求的四元组定义，且 Table 1 的"操作契约"也基本满足了五条契约要求。

**Section 4（第 61–79 行）**  
内 loop 被命名为"generation-time self-correction"，外 loop 被命名为"deployment-time self-improvement"，与约束 #3 完全吻合。Table 2 以四列展示了两层 loop 的同构结构（Level / Feedback oracle / Action space / Verdict role），这是约束 #3 的核心视觉锚点，这版已落实。

---

### 1.2 缺陷：叙事仍有"从 framework 退回到 task 描述"的局部滑点

**Section 1（第 9–17 行）问题最集中。**

第 9 行（Introduction 第 2 段）：
> "To build scene-aware object rotation pairs, each object must be rendered in a fixed environment with stable lighting, grounding, scale, viewpoint, and background consistency."

NARRATIVE_ARC 约束 #1 明确写道：`Section 1.1 仅作为痛点起点可以出现一次"旋转/azimuth/yaw/8 方位角"等术语；Section 2 与 Section 3 严禁使用这些术语`。

这版成品把 rotation/scene-aware/render 等任务特定术语渗入 Introduction 全段，并且在第 15 行（Introduction 第 5 段）的"main finding"部分再次出现"CLIP-I, Score_view, VIE Overall"等具体数字，然后在第 17 行提到 `inspect` 结果。

这意味着：Section 1（Introduction）就提前泄露了全部实验结果，违反了约束 #7"叙事必须严格按 小任务(1)→大框架(2-3)→回到小任务上的验证(4-5) 推进"的精神——Introduction 不应将结果摘要与动机陈述并列，而应以问题为主、结论为辅。

**后果**：读者在读完 Introduction 就已知道所有重要数字，Section 8 的结果部分因而缺乏"揭示"感。这在技术报告中可以接受，但 NARRATIVE_ARC 明确要求的叙事弧结构（小任务→大框架→验证）在这版的章节排布中被 "spoiled"。

---

## 二、叙事弧（小任务→大框架→回到验证）

### 2.1 章节顺序基本符合

整体章节顺序：Introduction → Motivation（Section 2）→ Framework（Section 3）→ Dual-Loop（Section 4）→ Pipeline Instantiation（Section 5）→ Case Study（Section 6）→ Validation Protocol（Section 7）→ Results（Section 8）→ Discussion（Section 9）→ Limitations（Section 10）→ Conclusion（Section 11）→ Reproducibility（Section 12）→ References（Section 13）。

这与 NARRATIVE_ARC 的 1→2→3→4→5 顺序在逻辑上对应，且 Section 2 的 Motivation 部分保持了任务中立的抽象层面。

### 2.2 约束 #1 违反情况（Section 2 与 3 是否出现 rotation/azimuth？）

**Section 2（第 25–35 行）**：严格检查后，Section 2 没有出现 azimuth/yaw/rotation editing 等任务特指术语，而是使用"object-level editing data"、"controlled changes and invariant preservation"等任务中立表述。这一点符合约束 #1。

**Section 3（第 37–47 行）**：Section 3 同样保持任务中立，没有使用 rotation/azimuth 术语。约束 #1 的 Section 2-3 部分通过。

**Section 4（第 61–79 行）**：同样未见 rotation/azimuth 术语，行文保持框架层面的泛化描述。约束 #1 的 Section 4 部分通过。

**真正的违反点在 Section 1（Introduction）**：如上所述，Introduction 大量使用了 rotation/scene-aware/object rotation 等术语。NARRATIVE_ARC 约束 #1 写的是"Section 1.1 仅作为痛点起点可以出现一次"——但 Introduction 这版已经不只出现一次，而是贯穿全段，且提前给出了所有实验结论数字。

---

## 三、R1 数字诚实度

### 3.1 数字核查（对照 CLAUDE.md）

| 指标 | CLAUDE.md 原始 | 成品第 129–139 行 Table 4 | 一致性 |
|------|:-:|:-:|:-:|
| PSNR exp5 | 16.63 | 16.63 | ✓ |
| PSNR R1 | 16.68 | 16.68 | ✓ |
| CLIP-I exp5 | 0.9050 | 0.9050 | ✓ |
| CLIP-I delta | +0.0449 | +0.0449 | ✓ |
| DINO delta | −0.0058 | −0.0058 | ✓ |
| FID exp5 | 50.83 | 50.83 | ✓ |
| FID delta | +5.10 | +5.10 | ✓ |
| Score_view delta | +0.0123 | +0.0123 | ✓ |
| VIE Overall delta | +0.0054 | +0.0054 | ✓ |

所有数字均准确，与 CLAUDE.md 完全一致。

### 3.2 正向与退化的披露平衡（约束 #5）

DINO 退化（第 135 行）：明确写出 `−0.0058`，解释列注明 "consistency regression"。  
FID 退化（第 136 行）：明确写出 `+5.10`，解释列注明 "distribution-quality regression"。  
DINO 强角度退化（第 143 行）：文字段落明确提到"45 degrees, with a delta of -0.00994 against a threshold of -0.008"。  

**结论**：约束 #5 完全满足，正向与退化均如实披露，无挑拣展示。

### 3.3 hybrid_score 数字（约束 #4 定量依据）

第 145 行：`All 20 newly added R1 objects have hybrid_score < 0.6, with a median around 0.46.`

约束 #4 要求"必须显式引用 20 个新物体 hybrid_score 全部 < 0.6 这一数字作为诊断结论的定量依据"。这版完全满足。

---

## 四、LoRA 定位（约束 #6）

**Section 7（第 108 行）**：
> "The report does not claim a new LoRA method; LoRA is used as an instrument for measuring whether the data engine produces useful supervision."

**Table 2 的 Verdict role 列（第 76 行）**中，外 loop 的 LoRA 被定位为"downstream LoRA probe and benchmark metrics"。

**Section 8 开篇（第 125 行）**：
> "The validation asks whether the constructed data changes downstream behavior..."

整个 Section 8 的 claim 没有一处写"LoRA 打败 baseline"，始终以数据引擎的诊断能力为主语。

**结论**：约束 #6 完全满足，LoRA 被正确写成 probe/instrument，不是头条结果。

---

## 五、R1=inspect 的诊断框架（约束 #4）

### 5.1 核心定位

Section 8.1（第 147–151 行）有专门的"Why `inspect` Is Not Failure"子节，这是这版成品相对前一版草稿的最大改进之一。该段核心句：
> "In this workflow, the same mixed result becomes actionable evidence."

解释逻辑完整：外 loop 找到有用方向 → 内 loop 质量门控不足 → inspect 是路由决策而非最终失败。

### 5.2 与 NARRATIVE_ARC 4.5 节的对比

NARRATIVE_ARC 4.5 节的核心 claim 是："`inspect` 裁决……是**外 loop 正确诊断出了内 loop 当前的瓶颈**——这恰是'两层 loop 都在工作'的证据"。

成品第 78 行（Table 2 下方的解释段）：
> "The two-loop design explains why `inspect` is a productive outcome. A mixed result is not merely a failed experiment; it is a signal that the outer loop found a useful direction but the inner loop did not yet enforce sufficient quality."

这与 NARRATIVE_ARC 的表述高度一致，措辞自然且论证完整。

### 5.3 轻微不足

NARRATIVE_ARC 4.5 节要求把诊断结论表述为"两个 loop 都在工作"，即同时证明外 loop 和内 loop 的功能。成品的 Section 8 分析虽然到位，但有一处细节：成品没有明确区分"外 loop 的 verdict logic 正常触发 regression guard"（外 loop 工作的证据）与"内 loop 质量门控未拦截低质量样本"（内 loop 的诊断对象），两者在文字上略有混淆。建议在"Why inspect is not failure"段落补一句显式说明"外 loop 的 regression guard 正常触发，证明外 loop 功能正常；问题在内 loop 未能过滤 hybrid_score < 0.6 的样本"。

---

## 六、格式与可发表性

### 6.1 与 REVIEW_OF_DRAFTS.md 对比

旧版草稿被批评的四大格式问题：
1. "Abstract + Title 立即锁定旋转" → 这版已修复（标题泛化，Abstract 以闭环过程为主）  
2. "Workflow 定义不形式化" → 这版已修复（Section 3 有四元组 + Table 1 操作契约）  
3. "References 为空" → 这版已修复（13 条参考文献，见下节）  
4. "inspect verdict 叙事模糊" → 这版已改善（专门子节 + 精确措辞）

### 6.2 章节结构完整性

| 标准 arxiv 章节 | 成品状态 |
|---|---|
| Abstract | ✓ 完整（第 3–5 行） |
| Introduction | ✓ 完整（Section 1，第 7–24 行） |
| Related Work / Motivation | 部分：Section 2（第 25–35 行）兼具动机与相关工作，但没有独立 Related Work 章节 |
| Method（Framework） | ✓ Section 3–4–5（第 37–104 行） |
| Experiments | ✓ Section 6–7（第 90–121 行） |
| Results & Analysis | ✓ Section 8（第 123–151 行） |
| Discussion | ✓ Section 9（第 153–161 行） |
| Limitations | ✓ Section 10（第 163–171 行） |
| Conclusion | ✓ Section 11（第 173–179 行） |
| References | ✓ 13 条（第 204–228 行） |
| Reproducibility | ✓ Section 12（第 181–203 行），但内容偏向实现细节，不是 arxiv 标准格式 |

**缺失的关键内容**：
- 没有独立 Related Work 章节。Section 2 仅引用了 3 篇（InstructPix2Pix、MagicBrush、DatasetGAN），远不够支撑"agentic dataset construction"这一领域定位。至少需要增加 agent-guided data generation、VLM-based quality control、self-evolving data pipelines 等相关方向的代表工作。
- 没有任何图表（Figure 0 占位）。Pipeline diagram、framework 架构图、loss curve、qualitative grid 全部缺失。Section 10 第 171 行自行承认"Qualitative grids, pipeline diagrams, failure-case figures, loss curves, and completed checkpoint hashes are not yet included"。这对 arxiv 技术报告是可接受的最低完成状态，但读者理解 framework 的门槛会大幅上升。
- 消融实验缺失（REVIEW_OF_DRAFTS.md 已指出）：这版没有修复，Section 9 讨论泛化时只有定性论证，没有任何对比实验。

### 6.3 Reproducibility 章节问题

Section 12（第 181–203 行）仍然更接近内部交接文档，而非 arxiv 标准的 reproducibility section。"Artifact Index"表格（第 197–202 行）直接暴露了本地路径 `arxiv_report/docs/`，不适合公开发布。

---

## 七、参考文献核验

### 7.1 可信度高的引用（8/12 条）

| 编号 | 引用 | arXiv ID | 可信度 |
|------|------|----------|--------|
| [1] | InstructPix2Pix | 2211.09800 | ✓ 真实，CVPR 2023 |
| [2] | MagicBrush | 2306.10012 | ✓ 真实，NeurIPS 2023 DB |
| [3] | DatasetGAN | 2104.06490（核验文件中） | ✓ 真实，CVPR 2021（注：正文引用无 arXiv ID，仅引用核验文件有） |
| [4] | Blender 官网 | 无 ID，官网链接 | ✓ 合理 |
| [5] | SAM | 2304.02643（核验文件中） | ✓ 真实，ICCV 2023（注：正文引用无 arXiv ID） |
| [6] | SAM 2 | 2408.00714 | ✓ 真实，2024 |
| [8] | VIEScore | DOI:10.18653/v1/2024.acl-long.663 | ✓ 真实，ACL 2024 |
| [9] | LoRA | 2106.09685 | ✓ 真实，2021 |
| [12] | SpatialEdit | 2604.04911 | ✓ 真实，2026（核验文件中可见） |

### 7.2 高风险引用（需重点核查）

**[7] Hunyuan3D 2.1（第 218 行）**  
`arXiv:2506.15442`，标题"Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material"。  
**风险**：arXiv 号 2506.15442 对应 2025 年 6 月发布，符合时间线。核验文件列出了完整长名单作者列表（Shuhui Yang 等），这一层面信息详细。但本报告的工作目录实际使用的是 2024 年 11 月的 Qwen-Image-Edit-2511 模型，而 Hunyuan3D 2.1 在 2025 年 6 月才发布——**如果研究开始于 2025 年前，实际使用的可能是 Hunyuan3D 1.0 或 2.0 版本**，引用 2.1 可能有虚引之嫌，需对照项目实际使用版本确认。

**[10] Qwen-Image Technical Report（第 224 行，致命风险）**  
`arXiv:2508.02324`，作者"Chenfei Wu et al."  
**严重问题**：  
- arXiv 号 2508.XXXXX 表明这篇文章在 2025 年 8 月才被提交。  
- 但 Qwen-Image-Edit-2511 这个模型的命名中"2511"代表 2025 年 11 月发布。  
- 更关键的是，Qwen-Image-Edit-2511 是 Qwen 团队在 Hugging Face 上发布的模型，从公开信息看其技术报告是"Qwen2-VL"系列（arXiv:2409.12191），不是"Qwen-Image Technical Report"。  
- "Chenfei Wu"是华为 NLP 研究员（NExT-GPT 等），不是 Qwen 团队核心人员。  
- **结论**：arXiv:2508.02324 极可能是幻觉 ID，与实际 Qwen-Image-Edit-2511 的技术报告不匹配。这是整个参考文献中最严重的问题，**必须核查后替换**。正确的引用应为 Qwen2-VL 的技术报告（arXiv:2409.12191，Wang Peng 等）。

**[11] Qwen-Image-Edit-2511 Hugging Face 链接**  
`https://huggingface.co/Qwen/Qwen-Image-Edit-2511`  
这是一个模型卡页面引用，不是学术文献。arxiv 技术报告中直接引用 HF 模型卡作为 Reference 可接受但不规范，建议改为引用对应的技术报告或直接在脚注中给出链接。

### 7.3 格式问题

- [3] DatasetGAN、[5] SAM 在正文的引用格式中没有 arXiv ID，而核验文件中有。建议在正文 References 中统一加入 arXiv 号，保持一致性。
- SAM 的 arXiv 号（2304.02643）未出现在正文 Reference 列表中，需补充。

---

## 八、中英文一致性

### 8.1 中文版整体质量

中文版（228 行）与英文版在章节结构、表格、数字、核心术语上高度一致，不是机翻产物，而是逐段意译。主要关键术语均保留英文原文（`workflow-as-skill`、`inspect`、`hybrid_score`、`verdict`、`LoRA`）并附中文解释，处理方式专业。

### 8.2 发现的不一致点

1. **中文版 Section 8 标题**（中文版第 150 行）："结果与诊断分析"，英文版（第 123 行）："Results and Diagnostic Analysis"——完全一致，无问题。

2. **中文版 Table 4 的"解释"列**（中文版 Section 8）与英文版 Table 4 的"Interpretation"列完全对应，数字一致，中文解释语义准确，无漏段。

3. **引用格式**：中文版 References 与英文版完全相同（保留英文），合理。

4. **轻微差异**：中文版 Section 2 末段（对应英文 Section 2 最后一段）有一处意译过度，英文原文"it treats data construction as a stateful closed-loop workflow rather than only a dataset-generation procedure"被翻成"这里关注的是将数据构建作为有状态闭环 workflow"，丢失了"rather than only a dataset-generation procedure"的对照关系，稍弱于原文。

5. **中文版无 Section 8.1 的子节标题**："为什么 `inspect` 不是失败"——中文版有对应标题（中文版 Section 8 第 8.3 节）。经核查，中文版有此子节，内容与英文版一致。无遗漏。

**总体结论**：中英文同步质量良好，无重大漏段。评分：9/10。

---

## 九、总评分与 Verdict

### 综合评分：6.5 / 10

| 维度 | 得分 | 说明 |
|------|:---:|------|
| Thesis 对齐度 | 8 | 核心 tagline 体现，框架结构清晰，有轻微叙事泄露 |
| 叙事弧约束遵守 | 6 | Section 2-3 通过；Introduction 违反约束 #1（提前全量泄露结论数字） |
| R1 数字诚实度 | 10 | 全部正向+退化数字如实披露，无挑拣 |
| LoRA 定位 | 10 | 明确写成 probe/instrument，未越权宣称 |
| inspect 诊断框架 | 8 | 专节说明，论证完整；两 loop 同时在工作的表述可进一步显式化 |
| 格式与可发表性 | 5 | 无独立 Related Work，无图，Reproducibility 不符合 arxiv 标准 |
| 参考文献核验 | 5 | [10] Qwen-Image TR 极可能是幻觉 ID，[7] Hunyuan3D 版本待确认 |
| 中英文一致性 | 9 | 高质量意译，无重大漏段 |

### Verdict：`major-revision`

不是 `reject-and-rewrite`（核心结构已立，数字已对），但有两个问题必须在公开发布前修复，否则有学术诚信风险：参考文献 [10] 幻觉 ID、以及格式完整性（无图）。

---

## 十、Top-5 必须修复项

### 修复项 1（致命）：参考文献 [10] 幻觉 ID 必须替换
**位置**：正文第 224 行，references_verification.md 对应条目。  
**问题**：`arXiv:2508.02324` 对应"Qwen-Image Technical Report"，作者"Chenfei Wu et al."，极大概率是幻觉——Chenfei Wu 不在 Qwen 团队，且 2508.XXXXX 时间戳晚于 Qwen-Image-Edit-2511 的发布（2511 即 2025 年 11 月）逻辑矛盾。  
**修复方向**：核查 Qwen-Image-Edit-2511 实际对应的技术报告。目前最可能的正确引用是 Qwen2-VL 技术报告（arXiv:2409.12191）或 Qwen2.5-VL（arXiv:2502.13923）。如无合适公开技术报告，可将此引用删除，仅在 Section 12 脚注中给出 Hugging Face 链接。

### 修复项 2（重要）：Introduction 叙事结构重组，去除提前数字泄露
**位置**：Section 1（第 9–17 行），具体是第 17 行的"main finding"段落提前给出了 CLIP-I、DINO、FID 等所有实验数字。  
**问题**：违反约束 #7（叙事顺序：小任务 → 大框架 → 回到验证），也违反了约束 #1 的精神（Introduction 的角色是提出问题和框架，而非展示结论）。  
**修复方向**：将第 17 行替换为一句简洁的结论性表述（"The main finding is diagnostic, demonstrating both useful weak-angle signal and an unresolved render-quality bottleneck"），删去具体数字和 hybrid_score 细节，这些数字留在 Section 8 再展开。contribution 列表可保留，但 point 3 应避免提及 "mixed R1 result"，改为"a single inspection round"。

### 修复项 3（重要）：添加独立 Related Work 章节（至少 8 篇）
**位置**：Section 2（第 25–35 行）当前兼任 Motivation 与 Related Work，仅引用了 3 篇。  
**问题**：缺少 agentic data generation、VLM-based quality control、self-evolving data pipelines 相关方向的覆盖，导致这篇技术报告看起来像孤立于领域的工程报告，而非学术贡献。  
**修复方向**：在 Section 2 和 Section 3 之间插入独立的 Related Work 章节，覆盖：(a) 指令引导图像编辑（已有 2 篇）；(b) 合成数据工厂方法（已有 1 篇，需增加 2–3 篇）；(c) agent 辅助数据工程；(d) VLM 质检与 evaluation-driven scaling；(e) 3D 资产自动化生成。

### 修复项 4（重要）：补充 Framework 图/Pipeline 图（即使是文字 ASCII 也好过没有）
**位置**：Section 3–5 全程缺失图示。  
**问题**：Section 10 自承 "pipeline diagrams... are not yet included"。一篇介绍两层 loop 框架的技术报告，没有一张图就发出去，审稿人和读者几乎无法在脑中构建 framework 的形象，Table 2 的两 loop 对比虽有帮助但不够。  
**修复方向**：最低限度在 Section 3 或 4 处插入一张文字式架构图（ASCII 方框或 Markdown 列表型伪图），标注 Request → [S1|S2|...|VLM Review|Verdict|Action] → Dataset；在 Section 4 的双 loop 说明中插入一张两 loop 流程图。图的缺失是这版报告与正式 arxiv 预印本之间最大的视觉差距。

### 修复项 5（中等）：Reproducibility 章节去除内部路径，改为标准格式
**位置**：Section 12（第 181–203 行），具体是第 197–202 行的"Artifact Index"表格。  
**问题**：`arxiv_report/docs/`、`arxiv_report/eval/`、`arxiv_report/code/` 这些路径是本地内部路径，不能出现在公开发布的技术报告中。  
**修复方向**：将 Artifact Index 删除或替换为"Code and data will be made available at [anonymous URL]"之类的标准占位符；将 Section 12 核心内容浓缩为超参表（rank=32, lr=1e-4, epoch=29）+ 数据集规格（N_train=305, N_val=49, N_test=56）+ checkpoint hash 占位符，按 arxiv 标准格式排版。

---

## 十一、与 REVIEW_OF_DRAFTS 对比（改善量化）

| REVIEW_OF_DRAFTS 旧草稿问题 | 旧版状态 | 新版状态 |
|---|---|---|
| Title 立即绑定 rotation | 未解决（旧标题直接含 rotation） | **已解决**（新标题用"Object-Level Image Editing"） |
| Abstract 过早提 rotation case study | 未解决 | **已解决**（Abstract 以闭环过程为主） |
| Workflow 定义缺形式化 | 未解决（纯描述） | **已解决**（Section 3 有四元组 + Table 1） |
| inspect verdict 叙事模糊 | 未解决 | **已解决**（专节"Why inspect is not failure"） |
| References 为空 | 致命（完全没有引用） | **已解决**（12 条引用，但 [10] 有幻觉风险） |
| 消融实验缺失 | 未提 | **仍缺失**，Discussion 只有定性论证 |
| Reproducibility 不标准 | 未提 | **仍有问题**（含内部路径） |
| 无图 | 未提 | **仍缺失** |
| Related Work 单薄 | 指出仅 3 篇 | **仍单薄**（Section 2 仅 3 篇，无独立 Related Work） |

---

## 文档元信息

- **审阅版本**：GPT-5.5 英文版 + 中文版（各 228 行）+ references_verification.md（103 行）
- **金标准参照**：NARRATIVE_ARC.md（172 行，9 条硬约束）+ REVIEW_OF_DRAFTS.md（397 行）+ CLAUDE.md R1 数字
- **生成时间**：2026-04-24
- **审阅者**：Claude Sonnet（claude-sonnet-4-6）
- **总评分**：6.5 / 10
- **Verdict**：`major-revision`
