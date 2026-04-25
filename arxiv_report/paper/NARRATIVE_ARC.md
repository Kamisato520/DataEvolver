# DataEvolver 技术报告叙事主线 (Narrative Arc)

> 本文件为 arxiv 技术报告的写作锚点。后续各章由不同 subagent 展开时，必须遵循此处锁定的主张、顺序、术语与数字。任何偏离都需先更新本文件。

---

## Thesis （整篇报告的灵魂）

> **这是整篇报告的灵魂。所有章节、所有 claim、所有数字都必须指向这句话；任何偏离都应视为写作错误而非叙事变体。**

**长版（中文）**：
> 合成数据集构建本质上不是一次性生成任务，而是一个需要观察、诊断、调参、重生成和验证反馈的闭环过程。我们将这个过程抽象为 **workflow-as-skill data engine**，并用 VLM + AI agent 替代部分人工 Blender 调参，在带背景物体旋转数据构建中验证其可行性。

**短版（英文，tagline）**：
> **We turn manual synthetic-data construction from a human-driven Blender tuning process into a VLM- and agent-guided self-evolving workflow.**

---

## 0. 一句话版本

**Claim**：我们把手工驱动的合成数据构建改写为一个由 VLM 与 AI agent 共同引导的自进化 workflow——*turning manual synthetic-data construction from a human-driven Blender tuning process into a VLM- and agent-guided self-evolving workflow*——并在最初触发这一观察的小任务（带背景物体 2D 新视角旋转）上拿到了可被框架自身诊断的验证结果。

> 10 秒电梯 pitch：小任务发现痛点 → 抽象出闭环过程 → workflow-as-skill 框架 → pipeline 实例化 → 回到小任务上验证 → R1 以 `inspect` 裁决，验证的同时也诊断出框架下一个瓶颈。

---

## 1. 起点：从一个小任务发现的痛点

### 1.1 具体场景：带背景物体的 2D 新视角旋转数据构建

**Claim**：一个看似边缘的小任务——为带背景物体生成 8 方位角配对的旋转视图——在实际执行时暴露了一个比任务本身严重得多的系统性障碍。

为了训练固定单轴旋转编辑模型，我们需要大量"同一物体、同一场景、不同 azimuth"的配对图像。现成真实数据集无法提供这种密度的角度覆盖，只能走合成路线：文生图 → 3D 重建 → Blender 渲染。然而真正耗费人力的不是模型或算法，而是 Blender 中每一个对象的光照、材质、视角和背景都需要人工反复调参；一次批量渲染出错，整个 pair 报废。临时补一小批新物体就意味着重新进 Blender 手动 vibe-coding 一遍。这是启发整份工作的起点，也是本节之后不再出现的具体任务术语。

### 1.2 手工合成数据集的三大痛点

**Claim**：这条链上的困难不是局部的工程瑕疵，而是在三个正交维度上同时失效。

第一，**可复用性差**：每一次渲染参数调整都是一次性的 ad-hoc 脚本，下一个物体/场景需要从头再来，没有可沉淀的中间产物或状态。第二，**质检不可扩展**：渲染成品的好坏高度依赖人工目视判断，规模到几百个物体就不再能全覆盖审查，低质量样本会悄悄污染训练集。第三，**无闭环**：数据被交付给训练/评测之后，评测结果不会回流指导下一批数据生产；下游模型的弱点只是变成工程师的 Slack 消息，不会变成数据层面的扩充策略。

### 1.3 抽象观察：合成数据构建本质上是一个闭环过程，而不是一次性生成任务

**Claim**：合成数据集构建的真正形态不是"生成一次就结束"的 pipeline，而是观察—诊断—调参—重生成—反馈的**闭环过程**；只要这条闭环里的任一决策节点依赖人类，整个系统就不可扩展。

这一观察把第 1.1–1.2 节的痛点从工程层面拔到范式层面：Blender 可以换成别的渲染器、旋转可以换成别的任务，但"生产—审查—诊断—再生产"这一结构不会消失，真正决定能否扩展的是这条闭环里有没有人类节点。"pipeline"这个词暗示单向流动，遮蔽了真正的难点；"workflow"才刻画了这种带反馈与重入的闭环结构。这也是为什么下一节的框架以 **workflow-as-skill** 而非 "data pipeline" 命名——名字里就要带闭环。只有把闭环里的人类节点换成可审计的 AI agent + 可机读的 feedback oracle，数据生产才可能真正扩展。

---

## 2. 框架：双 Loop AI-agent Workflow

### 2.1 设计原则：workflow-as-skill + agent + feedback oracle 的同构结构

**Claim**：既然数据构建本身就是"观察—诊断—调参—重生成—反馈"的闭环过程，那么合适的抽象就是 **workflow-as-skill data engine**——由 agent 执行、VLM 诊断、动作空间枚举、verdict 自动化；框架的唯一原语是"agent 在 feedback oracle 的监督下对一段 workflow 做带反馈的可恢复迭代"，两层 loop 只是这同一原语在不同时间尺度上的两次实例化。

具体来说，workflow-as-skill 意味着每一个生产阶段都被封装为一个带有显式输入/输出契约、状态持久化与可复现动作空间的 skill；skill 与 skill 之间通过有向依赖拼接成 pipeline。在每一个需要决策的节点，原先的"人"被换成两个角色：**AI agent**（Claude Code / Codex，承担动作选择与代码编辑）与 **feedback oracle**（VLM 或评测器，承担质量判据）。这种同构的好处是框架可以推广到任何"agent + 可机读 oracle"存在的任务，而不是只能做渲染或旋转。

### 2.2 Inner Loop: Generation-time Self-correction

**Claim**：内 loop 发生在数据被生产的那一瞬间，把原先"调参—试渲染—肉眼评估"的手工闭环替换为自动闭环。

形式是 `render → VLM review → action → re-render`。单个样本进入渲染阶段后，VLM（Qwen3.5-VL-8B）基于多维打分（外观一致性、视角正确性、前景/背景分离度等）给出 hybrid_score 与结构化的 review；agent 从离散动作空间（本工作中为 24 条可选修正动作）中选择并改写渲染参数，重新跑这一样本直到 hybrid_score 跨过门控阈值或步数用尽。这样"质检"就从 pipeline 末端的人工环节变成了 pipeline 中间的 oracle 节点。

### 2.3 Outer Loop: Deployment-time Self-improvement

**Claim**：外 loop 发生在数据集交付并被下游任务消费之后，把评测结果反向注入到下一批数据的生产计划里。

形式是 `eval → 诊断弱子群 → augmentation plan → re-build → re-train → re-eval`。下游训练得到的模型在基准上产生 per-子群指标，一个 compare 步骤识别出表现最差的子群（本工作中是几个特定角度），产出机器可读的 augmentation plan；plan 被送回内 loop 去生成有针对性的新样本，合并进增广数据集，重新训练并再次评测。外 loop 把"模型的弱点"翻译成"数据层面的扩充指令"，从而关闭了 1.2 节中提到的第三个痛点。

### 2.4 两层 Loop 的协作语义

**Claim**：两层 loop 共享同一种"agent + oracle + 动作空间 + 收敛判据"的抽象结构，仅在具体实例化上不同。

| 维度 | Inner Loop | Outer Loop |
|------|-----------|------------|
| 输入 | 单个样本与当前渲染参数 | 一整个训练轮次后的评测报告 |
| Agent | Claude Code / Codex（改渲染参数与场景脚本） | Claude Code / Codex（改扩充计划与训练配置） |
| Feedback oracle | VLM 多维打分（hybrid_score + 文本 review） | 评测脚本（per-子群指标 + verdict） |
| 动作空间 | 24 条离散修正动作 | 扩充物体集合、调整分布、可选调超参 |
| 收敛判据 | hybrid_score ≥ 门控 或 步数耗尽 | verdict ∈ {accept, inspect, reject} |
| 时间尺度 | 秒–分钟（每样本） | 小时–天（每训练轮次） |

这张表是 Section 2 的核心视觉锚点：它同时向读者展示"两层 loop 是同一个抽象"和"两层 loop 解决不同尺度上的问题"。

---

## 3. 例化：DataEvolver Data-Build Pipeline

### 3.1 Stage 1–5 概览

**Claim**：上一节的抽象框架被具体化为一条五阶段 pipeline，每一阶段既是一个 skill，也是一次可被内 loop 监督的生产单元。

pipeline 从 Stage 1 的文本扩充开始（Qwen-LM 基于一句数据集描述扩展出 N 条细粒度物体 prompt），进入 Stage 2 的 T2I 生成白底图（Qwen Image Edit，白底是下游分割的硬约束）、Stage 2.5 的 SAM3 前景分割，然后是 Stage 3 的 Hunyuan3D 重建（含 texture baking），Stage 4 的 Blender 场景渲染（用对象旋转而非相机轨道，把被渲染物体放进统一场景坐标下），最终在 Stage 5 由 Qwen3.5-VL-8B 执行 VLM 质检并驱动 2.2 节描述的内 loop。输出是一份 trainready 配对 CSV，可以直接被下游的训练脚本消费。

### 3.2 关键设计决策

**Claim**：四个看似局部的工程决策，都是为了让抽象框架在这条 pipeline 上真正闭环。

一是**场景感知渲染**而非孤立物体渲染：同一物体必须在同一场景下生成一组配对图，否则背景差异会污染外观一致性指标；二是**对象旋转而非相机轨道**：统一场景坐标下的 yaw 控制比相机运镜更容易被 VLM 审核和被下游模型学习；三是**VLM 24 动作空间**：把自然语言反馈压缩为可执行、可组合的动作是让内 loop 收敛的前提，也是 agent 不陷入漫无目的改写的关键；四是**外 loop 基于 per 子群指标识别弱点**：不是比较 overall 指标，而是在评测脚本里拆出 per-子群 delta，才能把"哪些样本缺"这件事机器化。

---

## 4. 在最初任务上的验证：8-Azimuth Rotation Editing

### 4.1 任务定义

**Claim**：我们回到第 1.1 节的小任务，让框架生产的数据去教一个现成模型学会做固定单轴旋转编辑。

具体地，给定一张"物体 @ yaw=0°"的源图与一个目标角度（来自 8 方位角离散集合），模型要预测该物体在目标角度下、同一场景中的新视图。评测使用 SpatialEdit-Bench（488 pairs = 61 objects × 8 angles，训练集只覆盖 45°–315° 共 7 角度，360° 仅出现在评测），指标是 PSNR / SSIM / LPIPS / CLIP-I / DINO / FID，以及 VIEScore（Score_view + Score_cons）。

### 4.2 数据集构成：exp5 baseline + R1 augmentation

**Claim**：baseline 与 R1 只在"数据集"这一个变量上不同，其它训练与评测配置严格对齐，使指标差直接归因于框架产出的增量数据。

exp5（ours_objinfo）是一个已经跑稳的 baseline：rank=32, lr=1e-4, epoch 29, Prompt v3。外 loop 在这份 baseline 的评测结果上识别出几个弱子群，并驱动内 loop 在 wwz 服务器上新产出 20 个物体在弱角度上的 60 个训练 pair，合并后得到 305 train / 49 val / 56 test 的增广数据集，训练到同样的 epoch 29，其它配置不变。

### 4.3 训练探针角色（LoRA on Qwen-Image-Edit-2511）

**Claim**：LoRA 在这里是 **probe（探针），不是主结果**；framework 的 claim 是"能提供有效监督"，LoRA 只是让这个 claim 变得可测。

我们训练的是 Qwen-Image-Edit-2511 上的 LoRA，而不是从零训练模型——选择这一探针是因为它对数据质量高度敏感、训练代价低、可在同卡数/同 epoch 上重复跑；如果框架产出的数据是有效的监督信号，探针应该给出可读取的 delta。反过来，如果探针在某些子群上退化，外 loop 必须能把这种退化诊断成"下一步要修什么"——这直接连到 4.5 节。

### 4.4 R1 主要数字（3 个正向 + 2 个退化）

**Claim**：R1 同时观察到 3 个正向指标与 2 个真实退化，且正向信号恰好集中在外 loop 定向增广的弱子群上。

正向信号：CLIP-I 从 0.9050 提升到 0.9499（+0.0449），Score_view 从 0.7705 提升到 0.7828（+0.0123），PSNR 从 16.63 提升到 16.68（+0.05，且在 7/8 子群上单调提升）。更重要的是，Score_view 的提升集中在被外 loop 识别为弱的 135°（+0.049）、180°（+0.049）、315°（+0.033）三个子群，证明"定向增广 → 语义视角正确性提升"这条链是通的。

退化信号：DINO 从 0.8895 下降到 0.8837（−0.0058），FID 从 50.83 上升到 55.93（+5.10），Score_cons 从 0.9709 微降到 0.9676（−0.0033）。DINO 的退化发生在多个子群（包括训练集里本来就强的角度），FID 的退化反映了生成图像分布偏移。这两个退化必须被诚实披露，不能只引用 CLIP-I。

### 4.5 R1=inspect 的诊断式解读

**Claim**：R1 收到 `inspect` 裁决不是实验失败，也不是实验成功，而是**外 loop 正确诊断出了内 loop 当前的瓶颈**——这恰是"两层 loop 都在工作"的证据。

裁决逻辑是这样的：外 loop 的 compare 脚本看到 strong-angle DINO 在多个子群退化，触发 regression guard，拒绝把 R1 数据集一键转正为新 baseline，改为标记 `inspect`。进一步回溯，这 20 个新物体的 hybrid_score **全部 < 0.6**（远低于 golden-config 约 0.78 的目标）——也就是说，内 loop 的质量门控在这批样本上实际上没有卡住低质量输出，污染了外观一致性。换句话说，**外 loop 识别出内 loop 的质量门控不足**，并把这件事以 `inspect` 的形式显式地汇报给操作者，而不是静默地接受一个在某些维度退化的新 baseline。这正是框架在第 2 节宣称的"诊断即贡献"在 R1 这一轮上的具体兑现。

---

## 5. 框架带来了什么（Takeaway）

### 5.1 双 loop 的同构结构让 framework 可推广到新任务

**Claim**：因为两层 loop 都是"agent + oracle + 动作空间 + 收敛判据"的实例，把框架迁移到新任务只需要替换这四个坑位，而不需要重写 pipeline 的控制流。

对新任务，只需：定义一个新的 VLM review 维度集合（替换内 loop 的 oracle）、定义一个新的 per-子群评测脚本（替换外 loop 的 oracle）、枚举该任务相关的动作空间，其余状态持久化、verdict 逻辑、可复现性机制均可复用。

### 5.2 failure 被转为 feature：诊断即贡献

**Claim**：传统 ad-hoc 数据生产流程里，一次失败的数据集迭代只是"浪费的一天"；在本框架里，一次失败的迭代会被外 loop 翻译成下一批数据生产的具体指令。

R1 的 `inspect` 裁决就是一个落在实际实验中的例证：它不仅指出"不要把这份数据转正"，还指出了"质量门控阈值需要上调 / hybrid_score < 0.6 的样本不应进入训练集"这一层可执行的修复方向。这把"负结果"本身也变成可累积的研究资产。

### 5.3 未来：真实数据集的扩展路径

**Claim**：本工作在纯合成数据集上验证了框架可用性，下一步是把内 loop 的 oracle 扩展到混合数据（真实采集 + 合成增广），让外 loop 能跨数据源做弱子群诊断。

路线上包含三个方向：(a) 用 VLM oracle 审核真实采集样本，把"真实数据也可能被坏样本污染"纳入同一套质量门控；(b) 把外 loop 的 augmentation plan 扩展到"指定采集列表"而不仅仅是"指定合成列表"；(c) 在 R2 中先把 R1 识别出的质量门控问题修掉（阈值与 24-action 设计调整），再进入真实数据扩展。

---

## 6. 叙事约束清单（写作时的硬约束）

> 以下为 Section 1–5 各章写作 subagent 的硬性锚点。违反任何一条都必须先回到本文件修改并说明理由。

1. **术语隔离**：Section 1.1 **仅作为痛点起点**可以出现一次"旋转 / azimuth / yaw / 8 方位角"等术语；Section 2 与 Section 3 严禁使用这些术语，框架描述必须保持任务中立；这些术语在 Section 4 之后才能重新出现。
2. **抽象上移**：Section 1.3 必须把 Blender 痛点抽象为"human-in-loop 是合成数据生产不可扩展的根本瓶颈"这一范式级观察，不得停在工程吐槽层面。
3. **两层 loop 命名锁定**：内 loop 必须写作 **generation-time self-correction**，外 loop 必须写作 **deployment-time self-improvement**；两者必须被明确描述为"agent + feedback oracle"的同构结构。
4. **R1 verdict 定位**：R1 的 `inspect` 不能写成"实验失败"也不能写成"实验成功"，必须写成"framework 诊断出下一瓶颈"。必须显式引用"20 个新物体 hybrid_score 全部 < 0.6"这一数字作为诊断结论的定量依据。
5. **数字一致性**：所有指标必须以 `compare_report.json` / `R1_vs_exp5_spatialedit_bench.md` 为准。必须同时披露正向（CLIP-I +0.0449、Score_view +0.0123、PSNR +0.05）与退化（DINO −0.0058、FID +5.10、Score_cons −0.0033）；禁止只挑正向指标。
6. **LoRA 角色定位**：LoRA 必须被显式称作 **probe（探针）**，不是主结果；framework 的主张在于"能提供有效监督"，而不是"LoRA 打败 baseline"。
7. **顺序锁定**：叙事必须严格按 "小任务（1）→ 大框架（2–3）→ 回到小任务上的验证（4–5）" 推进，任何章节之间的前后顺序调整都必须同步更新本文件的约束清单。
8. **版权与引用**：所有外部方法描述（T2I、SAM3、Hunyuan3D、Qwen-VL 等）不得长段落复述其原论文，必须以短引用 + 参考文献条目方式处理；Section 2 的引用必须在正文出现前补齐，不得留"TBD"。
9. **Thesis 锁定**：所有章节的 claim 必须可追溯到顶部的 Thesis 卡片（中文长版 + 英文 tagline）。写作时若某节论述偏离"观察—诊断—调参—重生成—反馈的闭环"这一核心叙述，应立即回检并修正，或先更新 Thesis 卡片后再展开。
