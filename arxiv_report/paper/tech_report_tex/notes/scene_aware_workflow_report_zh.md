# 从人工 Blender 调参到自演化数据构建：面向物体级图像编辑的 VLM- 与 Agent-Guided Workflow-as-Skill 框架

## 摘要

物体级图像编辑的合成数据集构建常被看作前向生成 pipeline，但实际过程更接近观察、诊断、参数调整、重生成和验证反馈组成的闭环。本文将这一过程抽象为 workflow-as-skill 数据引擎：一种可复用、有状态、可检查的框架，由分阶段产物、VLM/CV 审核、AI agent 动作、下游探针和显式 verdict 驱动。该框架包含用于生成期自校正的内循环，以及用于验证期自改进的外循环。本文在原始动机任务上验证该想法，即场景感知物体旋转数据构建；workflow 生成 train-ready 配对数据，并同时暴露出有用的弱角度信号和未解决的渲染质量瓶颈。因此，`inspect` verdict 不是失败，而是一个诊断结果，说明继续扩容前必须强化质量门控。

## 1. 引言

本文从一个小但持续存在的合成数据构建瓶颈出发。为了构建场景感知物体旋转 pair，每个物体都必须在固定环境中渲染，并保持光照、接地、尺度、视角和背景一致性。实际操作中，这需要反复进行 Blender-based 调参 [4]：渲染样本、检查失败、诊断问题来自摆放、光照、相机取景、材质质量还是重建伪影，调整参数或脚本，然后重新渲染。瓶颈并不只是 Blender 不方便。更深层的问题是，人工合成数据构建本身已经是一个 human-in-the-loop 的闭环过程，只是这个闭环通常是隐式的、低效的、难以扩展的。

当目标只是构建一个小批量但任务特定的数据集时，这个痛点会显著放大。用户可能只是想增加一批新物体，或快速构建一个新的物体级编辑任务数据集，却仍然需要渲染经验、反复试错，以及有时通过 Blender MCP-style 工具或直接代码编辑进行探索式代码级调参。缺少显式 workflow 时，同类失败会在不同物体和数据轮次中反复出现。更糟糕的是，如果下游训练发生回归，很难追踪原因到底是坏资产、差渲染、弱指令、split 问题还是训练配置。

核心观察是，合成数据集构建不应被建模为一次性 pipeline。它是一个迭代 workflow：观察、诊断、调整、重生成和验证。本文提出 workflow-as-skill 数据引擎来显式化这一循环。该引擎将数据集请求转化为分阶段产物、审核信号、反馈动作和验收结果。VLM/CV 审核和 AI coding agent 替代部分人工 Blender 调参过程，而下游训练与评测在数据轮次层面提供反馈。

本文在原始动机任务上评估该框架：场景感知物体旋转数据构建。该任务适合作为验证案例，因为它要求稳定背景上下文、物体身份保持、视角控制、光照与接地一致性。它还迫使我们区分物体旋转和 camera-orbit multiview rendering。这个案例不是通用框架本身，而是一个具体验证场景，用来测试 workflow 是否能生成有效的 train-ready 数据，并诊断自身瓶颈。

本报告的核心发现是诊断性的：workflow 揭示出有用的弱角度信号，同时也暴露出尚未解决的渲染质量瓶颈，说明两层 loop 都在主动塑造结果。具体数字推迟到 Section 8 呈现。

本文贡献包括三点：

- 将自主视觉数据集构建从一次性生成 pipeline 重新表述为闭环 workflow-as-skill 数据引擎。
- 定义一种双循环设计，其中 VLM/CV 反馈和 AI-agent 动作用于生成期自校正，下游探针用于验证期自改进。
- 在场景感知物体旋转编辑上实例化该框架，说明语义泛化提升与一致性退化同步出现的混合结果可以成为可行动的诊断证据，而不是被浪费的试错。

## 2. 动机：从人工 Blender 调参到 Agentic 数据集构建

动机中的工程问题是重复渲染调参。场景感知合成数据集不能只通过是否存在图像文件来判断。物体必须完整、可见、接地、光照合理、尺度正确，并与目标指令一致。当这些条件中的任何一个失败时，下一步通常不是单一确定性修复。构建者必须检查渲染图，推断失败模式，决定是否调整光照、物体尺度、相机距离、支撑平面处理、材质设置或过滤规则，然后重新生成样本。

这个模式说明了为什么临时 pipeline 脆弱。脚本序列可以生成图像、mask、mesh、render 和 train pair，但它本身并不编码一个产物为什么值得信任。它也没有保留足够状态来解释下游回归。如果模型在训练后改善了视角对齐却损失了身份一致性，系统需要知道冲突来自哪个数据轮次、哪个物体子集、哪个渲染质量信号或哪个验证子组。

Agentic 数据集构建正是对这种闭环性质的回应。目标不是移除所有人工判断，也不是声称 agent 可以替代专家数据集设计。目标更窄也更实际：把观察-诊断-调整循环显式化，使 VLM/CV 反馈和 AI coding agent 能够承担一部分常规调参，而人类在 verdict 边界上保留监督。这把人工 Blender 迭代转化为带有持久产物和有界动作的可检查 workflow。

同样的动机并不限于 Blender 渲染。物体级编辑数据通常都要求受控变化和不变量保持，无论数据源是合成渲染、真实采集图像还是两者混合。因此，尽管第一个验证案例来自场景感知物体旋转，本文框架在抽象层面保持任务中立。

本文建立在指令引导图像编辑和编辑数据集的任务背景上；InstructPix2Pix 和 MagicBrush 将自然语言指令与 source-target 图像编辑样例联系起来 [1,2]。它也与 DatasetGAN 这类合成数据工厂工作相关，后者使用生成模型以较少人工标注成本产生带标签视觉数据 [3]。本文重点不同：这里关注的是将数据构建作为有状态闭环 workflow，而不只是一个数据生成过程。


## 3. 相关工作

**指令引导的图像编辑。** InstructPix2Pix [1] 和 MagicBrush [2] 确立了基于自然语言的图像编辑范式，二者都依赖人工构建的配对监督数据。本文关注的是上游问题：这些配对数据如何被生产出来。

**合成数据工厂。** DatasetGAN [3] 和 OmniGen [13] 表明生成模型可以制造任务特定的训练数据。我们将这一思路扩展为带有显式 verdict 门控的有状态 workflow，而非一次性通过的 pipeline。

**VLM 作为评判与反馈驱动的 scaling。** ImageReward [15] 使用学习型奖励对生成图像打分；VIEScore [8] 把 VLM 作为结构化评估器。本工作进一步推动这一方向：VLM 的 verdict 同时驱动内 loop 的重生成动作和外 loop 的数据扩充决策。

**3D 感知的视角合成。** Zero-1-to-3 [14] 通过渲染 Objaverse 资产训练视角扩散模型。我们的 pipeline 同样采用先渲染再训练的配方，但插入了 VLM 质量门控，避免低质量 3D 重建污染下游监督。

**自改进训练 pipeline。** STaR [16] 和 Self-Rewarding Language Models [17] 在 NLP 中推广了迭代式自改进 loop。本文的双层 loop 设计将同样的模板移植到多阶段视觉数据引擎，判官是多模态 VLM，动作是具体的 pipeline 阶段操作而非 token 级生成。

## 4. Workflow-as-Skill 数据引擎

本文提出的系统是 workflow-as-skill 数据引擎。它用可复用、有状态、可检查的过程替代松散脚本链，将数据集请求转化为分阶段产物、反馈信号、动作和 verdict。在当前表述中，skill 可以写为：

```text
Skill = (Stages, Controller, Review, Verdict)
```

`Stages` 是有序构建模块及其必须产生的产物。`Controller` 是 AI agent 或 workflow manager，负责追踪状态、路由样本、编辑脚本、调整参数并决定下一阶段。`Review` 包含自动 CV 信号、VLM 评估、人类可读 trace 和下游验证指标。`Verdict` 将观察到的证据映射为 `continue`、`inspect`、`regenerate`、`reject`、`stop_or_revert` 或 `no_signal` 等动作。

这种形式化是轻量的。它并不是要证明数据集正确性，而是要明确操作契约：每个阶段输出可检查产物，每个产物都挂接到持久状态，每个审核信号都可以影响路由，每个构建轮次都以避免含混解释的决策结束。

**表 1. Workflow-as-skill 数据引擎的操作契约**

| 组件 | 契约 |
|---|---|
| 阶段输出 | 构建模块输出 prompt、mask、mesh、render、review trace、train pair、checkpoint 和 comparison report，这些产物可以被检查或复用。 |
| Controller 动作 | Controller 路由样本、调整参数、必要时编辑 workflow 代码、触发重渲染或重生成，并跨轮次管理状态。 |
| 审核信号 | 质量通过自动 CV 信号、构建期 VLM 反馈和训练后下游 benchmark 行为共同评估。 |
| 反馈边界 | 更新被限制在重渲染、重生成、过滤、拒绝或延迟接受等有界动作内。 |
| Verdict 逻辑 | 只有在有用提升未触发 regression guard 时才接受；混合证据进入 `inspect`。 |

该抽象的价值在于可追踪性。数据引擎不应只生成样本，还应解释为什么这些样本足够可信、可以进入训练。持久状态使失败可追踪，审核信号使质量可见，verdict 使迭代决策显式化。下游训练仍然重要，但它作为数据引擎的验证探针，而不是主要方法本身。


图 1（文字示意）：Workflow-as-Skill 抽象。

```
  Request --> [ Stage_1 -> Stage_2 -> ... -> Stage_K ]
                           |
                           v
                   [ VLM Review ]
                           |
                           v
                   [ Verdict in {accept, regenerate, reject} ]
                           |
             +-------------+-----------+
             v             v           v
         accept        regenerate    reject
      -> Dataset    -> Action a in A -> discard sample
                      -> Return to earlier stage
```

## 5. 双循环自演化数据集构建

该 workflow 包含两个相互连接的循环。内循环发生在生成过程中，外循环发生在训练和评测之后。两个循环共享同一种结构：agent 接收来自 oracle 的反馈，在有界 action space 内行动，并根据收敛或 verdict 标准停止。它们主要区别在时间尺度和作用对象。

内循环是生成期自校正。样本被渲染后，由 VLM 和 CV 信号审核光照、接地、视角、尺度、物体完整性、材质合理性和背景一致性。Controller 使用这些反馈调整 Blender 参数、修改渲染脚本、改善资产摆放、改变相机或取景设置、过滤低质量样本，或触发重生成。这个循环替代了部分原本需要专家反复查看和改代码的人工 Blender 调参过程。

具体而言，agent action space 保持务实且有边界。它包括光照与曝光调整、相机和取景修正、物体尺度与位置修正、support-plane grounding、render filtering、资产重生成，以及当重复失败表明渲染规则或 controller 行为需要改变时进行的有限脚本编辑。这些动作被约束在可追踪范围内，使 workflow 能够改进样本，而不会把每次失败都变成不可追溯的手工重写。

外循环是部署期自改进。一个数据轮次导出后，系统训练下游探针、运行推理、计算指标、识别弱子组，并将结果转化为增强或过滤计划。该计划反馈到下一轮数据构建。在当前案例中，弱角度行为可以触发有针对性的新物体加入，而 regression guard 可以在新数据引入有害噪声时阻止自动接受。

**表 2. 数据引擎中的内循环与外循环**

| 循环 | 层级 | 反馈 oracle | Action space | Verdict 作用 |
|---|---|---|---|---|
| 内循环 | sample/render 层级 | 对生成产物的 VLM/CV 审核 | 调整渲染参数、编辑脚本、重渲染、重生成、过滤 | 判断样本或资产是否可用 |
| 外循环 | dataset/training-round 层级 | 下游 LoRA 探针和 benchmark 指标 | 增加目标数据、过滤子集、修改质量门控、检查或回退轮次 | 判断数据轮次是否应继续 |


图 2（文字示意）：两层同构 loop。

```
  内 loop（生成时）：
     渲染 -> VLM -> verdict -> action -> 重渲染 -> ...

  外 loop（部署时）：
     训练 -> 评测 -> compare.py -> verdict -> dataset_feedback_plan -> 数据扩充 -> 训练 -> ...

  共同结构：(agent, feedback oracle, action space, verdict)。
```

双循环设计解释了为什么 `inspect` 是有生产力的结果。混合结果不只是失败实验；它意味着外循环找到了有用方向，但内循环尚未实施足够强的质量约束。因此正确响应不是放弃框架，而是在继续扩容前改进内部质量门控。

## 6. Data-Build Pipeline 实例化

Data-build pipeline 将 workflow 抽象实例化为一组产生 artifact 的阶段。流程从物体概念扩展开始，因为数据引擎需要显式物体身份、描述和合成 prompt，才能控制后续视觉生成。白底物体生成随后产生隔离物体图，这比带有场景杂乱背景的图像更适合分割和重建。

前景分割将每张生成物体图转换为适合图生 3D 重建的 RGBA 资产。这个阶段与 SAM-family promptable segmentation 方法相关 [5,6]，但本文把分割视为实现组件，而不是新的分割贡献。重建 mesh 让 workflow 获得可插入场景、可变换、可接地、可打光、可从受控视角渲染的对象；Hunyuan3D 2.1 等 image-to-3D 资产生成系统为这一阶段提供相关技术背景 [7]。这些阶段在概念框架中不是彼此独立的脚本，而是定义 controller 可以检查和复用内容的 artifact 契约。

场景感知渲染随后把重建物体放入固定 Blender 环境中 [4]。这个阶段最集中地体现了原始人工调参痛点。渲染必须保留场景，使用稳定光照，避免漂浮或裁切，并让物体保持合理尺度。VLM/CV review 提供构建期反馈，而 controller action 调整参数、触发重渲染，或将样本标记为重生成/过滤。

当可用的 canonical source state 被选出后，workflow 导出目标视图并构建 train-ready source-target pairs。每个 pair 记录 source image、target image、instruction、target rotation、object identifier、object name 和 prompt version。该 schema 有意贴近下游图像编辑接口：source image 与 instruction 形成输入，target image 提供监督编辑目标。这个通用 pair schema 在水平旋转案例中变得具体：source 是 canonical front view，每个 target 是同一场景中的受控 object-yaw view。这使数据集可以被 LoRA 验证探针直接使用，同时保留可检查的构建轨迹。

## 7. Case Study：场景感知物体旋转

本文回到原始动机任务，验证 workflow 是否能生成有用的 train-ready 数据并诊断自身瓶颈。场景感知物体旋转是合适的第一个案例，因为它要求稳定背景上下文、物体身份保持、视角控制、光照一致性和接地一致性。它还暴露了一个重要歧义：物体旋转不同于 camera-orbit multiview rendering。

该任务特定的渲染契约是：

```text
canonical yaw000 state -> rotate object yaw -> keep scene and camera fixed
```

这条契约属于 case study，而不是通用框架。它避免两类失败模式。第一，如果每个目标角度独立选择 best render，会引入材质、光照、尺度或接地的跨角度漂移。第二，如果移动相机，任务会变成 camera-orbit multiview rendering。通过固定场景和相机、只旋转物体 yaw，pair 更直接地编码物体级旋转编辑。

Baseline 数据集使用 canonical front view 作为 source，非零水平方位视图作为 targets。每个训练物体贡献七个目标视图，35 个训练物体产生 245 个训练 pair。验证和测试划分采用 object-disjoint，分别包含 49 个验证 pair 和 56 个测试 pair。R1 中，评测反馈选择弱角度，新增物体只进入训练集。该轮在 90、180 和 270 度贡献 60 个弱角度 pair，将训练集扩展到 305 个 pair，同时不污染验证或测试物体。

因此，该 case study 连接了两个循环。内循环构建并审核可渲染物体数据。外循环使用验证反馈决定新数据应加在哪里。结果是一个具体的 train-ready 数据轮次，可以在不改变 held-out protocol 的情况下被评估。

## 8. 验证协议

验证要回答构建数据是否按预期改变下游行为，以及 workflow 是否能在回归出现时阻止过早接受。协议使用 LoRA fine-tuning [9] 在 Qwen-Image-Edit-2511 上作为下游探针；Qwen-Image 技术报告和 Qwen-Image-Edit-2511 model card 记录了该探针使用的图像编辑 backbone [10,11]。本文不声称提出新的 LoRA 方法；LoRA 只是用于测量数据引擎是否产生有效监督的工具。

exp5 baseline 是 R1 的比较点，因为反馈闭环状态使用 object-info prompt checkpoint。Baseline 和 R1 数据集遵循上文描述的 pair schema 和 object-disjoint split。物体划分在轮次之间冻结，因此验证或 benchmark 行为变化可以归因于训练数据干预，而不是泄漏或 split 漂移。

已跟踪训练探针使用 rank 32、学习率 1e-4、30 个训练 epoch，并统一使用 epoch 29 checkpoint 进行比较。Prompt 使用 train-ready 数据中的 instruction 字段，以视角语言要求模型将物体从 front view 旋转到目标 side/back view。这些设置用于复现，但主要主张仍然关于数据构建和诊断。

**表 3. 验证配置**

| 配置 | 训练物体 | Train pairs | Val pairs | Test pairs | 数据干预 |
|---|---:|---:|---:|---:|---|
| exp5 baseline | 35 个原始训练物体 | 245 | 49 | 56 | baseline object-info prompt checkpoint |
| v2 Scaling R1 | 35 个原始物体 + 20 个 train-only 物体 | 305 | 49 | 56 | 在 90、180、270 度弱角度新增 60 个 pair |

外部评测使用 SpatialEdit-Bench [12]，共 488 个 rotation pair，覆盖 61 个物体和八个角度槽位。指标包括用于像素保真度的 PSNR、用于结构相似度的 SSIM、用于感知距离的 LPIPS、用于语义图像相似度的 CLIP-I、用于表征相似度的 DINO、用于分布质量的 FID，以及用于 VLM 视角/一致性判断的 Score_view、Score_cons 和 VIE Overall。VLM-based scoring 与 VIEScore 等 visual-instruction evaluation 工作相关 [8]。角度槽位报告遵循 `compare.py` 中 index 0..7 到 45、90、135、180、225、270、315 和 360 度的映射。

## 9. 结果与诊断分析

验证揭示的是混合但有信息量的结果。R1 改善了若干语义和视角相关指标，说明有针对性的弱角度数据按预期改变了模型行为。同时，一致性和分布指标回归，阻止了自动接受。这正是双循环框架应有的行为：外循环识别有用信号，而 verdict 暴露出内循环渲染质量门控不足。

**表 4. exp5 baseline 与 v2 Scaling R1 的 SpatialEdit-Bench 整体指标**

| 指标 | exp5 baseline | v2 R1 | Delta | 解释 |
|---|---:|---:|---:|---|
| PSNR ↑ | 16.63 | 16.68 | +0.05 | 轻微保真度提升 |
| SSIM ↑ | 0.7296 | 0.7310 | +0.0014 | 轻微结构提升 |
| LPIPS ↓ | 0.2564 | 0.2546 | -0.0018 | 轻微感知提升 |
| CLIP-I ↑ | 0.9050 | 0.9499 | +0.0449 | 明显语义提升 |
| DINO ↑ | 0.8895 | 0.8837 | -0.0058 | 一致性回归 |
| FID ↓ | 50.83 | 55.93 | +5.10 | 分布质量回归 |
| Score_view ↑ | 0.7705 | 0.7828 | +0.0123 | 视角提升 |
| Score_cons ↑ | 0.9709 | 0.9676 | -0.0033 | 轻微一致性回归 |
| VIE Overall ↑ | 0.8649 | 0.8703 | +0.0054 | VIE 综合小幅提升 |

正向指标说明弱角度增强不是任意噪声。CLIP-I 明显提升，Score_view 上升，VIE Overall 增加。PSNR、SSIM 和 LPIPS 整体也朝期望方向移动，尽管幅度较小，不应过度解释。这些结果支持外循环继续寻找有针对性的弱角度数据。

回归指标解释了为什么该数据轮次不能被视为干净成功。DINO 下降，FID 变差，Score_cons 轻微下降。比较报告还记录了 45 度强角度上的 DINO regression，delta 为 -0.00994，阈值为 -0.008。这些回归重要，因为物体级编辑不仅要达到正确语义或视角，还要保持身份和外观。

渲染质量诊断指出了直接瓶颈。20 个 R1 新增物体全部 `hybrid_score < 0.6`，中位数约 0.46。这说明外循环找到了有用扩容方向，但内循环允许低质量合成物体进入训练干预。因此合适 verdict 是 `inspect`，而不是 `continue`。

### 为什么 `inspect` 不是失败

`inspect` verdict 是 workflow 的贡献，而不是 workflow 失败的迹象。在传统临时数据 pipeline 中，混合数据轮次往往只是被浪费的时间：有些指标提升，有些指标回归，构建者缺少下一步该修什么的明确线索。在这个 workflow 中，同样的混合结果变成可行动证据。系统指出弱角度扩容有潜力，但在继续扩容前必须改进渲染质量门控、资产重生成和类别感知渲染先验。

这种解释保留了证据的两面。R1 不是完全成功，因为一致性和分布指标回归。它也不是失败，因为它识别了有用的数据方向和具体质量瓶颈。Workflow 的价值在于把这种含混结果转化为被路由的下一步动作。

## 10. 讨论：泛化与边界

Framework-first 视角把数据构建方法从旋转案例中分离出来。更广泛的目标是为需要受控变化和不变量保持的物体级编辑任务提供可复用数据引擎。具体 yaw 契约不会自动迁移到其他任务，但 workflow 结构可以迁移：分阶段 artifact 构建、审核、有界反馈动作、下游验证和显式 verdict。

这种泛化必须谨慎表述。该 workflow 不会自动解决缩放、重定位、材质编辑、物体插入、姿态编辑或真实数据收集。每个任务都必须定义自己的 change/invariance contract、artifact schema、质量信号、action space 和验收逻辑。框架提供的是一种有纪律的表达和检查这些要求的方式。

下一条扩展路径是混合或真实数据构建。未来版本可以用 VLM 反馈审核真实采集样本，识别覆盖缺口或质量问题，并把下游弱点转化为合成增强计划或真实数据采集计划。同样的质量门控逻辑可以跨数据源使用。本文不声称系统已经解决真实数据构建；这里只指出同一闭环抽象的合理扩展方向。

最重要的实践教训是，质量门控必须内置于数据引擎。如果低质量样本仅仅因为被生成就能进入训练，扩容就是不安全的。下一版 workflow 应把渲染质量、资产完整性、视角-指令一致性和类别感知渲染先验作为一等验收条件。

## 11. 局限性与未来工作

当前验证是一个初始的单轮 `inspect` 案例。它说明框架能暴露有用信号和有害噪声，但不能证明闭环系统已经完全解决，或已经在多轮扩容中稳定可靠。后续需要在更强的内循环质量门控下继续实验，以测试已识别瓶颈能否被消除。

合成资产质量是当前直接边界。R1 物体低于期望 hybrid-score 阈值，使该轮作为诊断案例很有价值，但作为干净 scaling 结果仍较弱。下一步应在 export 或 train-set merge 前加入渲染质量门控，重生成或拒绝低质量合成资产，并改进类别感知渲染先验，使新物体类别不会继承不合适的光照、材质或尺度假设。

该框架还需要在场景感知旋转之外验证。未来任务不应假设旋转特定渲染契约可以直接迁移。每个任务都应定义自己的变换、不变量、审核信号和 verdict 逻辑。混合合成/真实数据设置尤其重要，因为它将测试同一 workflow 是否能同时指导合成增强和真实数据采集。

面向公开技术报告发布的若干 artifact 仍不完整。定性网格、pipeline diagram、failure-case figure、loss curve 和完整 checkpoint hash 尚未包含。角度标签一致性也需要注意，因为历史笔记对部分角度槽位使用了不同标签；最终 per-angle 表格应由单一权威脚本重新生成。

## 12. 结论

本文提出了一种用于物体级图像编辑自主合成数据集构建的 workflow-as-skill 框架。该框架源自一个具体的人工构建痛点：场景感知合成数据需要反复观察、诊断、Blender 参数调整、重生成和验证。通过显式化这一循环，数据引擎将人工调参转化为包含分阶段产物、VLM/CV 审核、AI-agent 动作、下游探针和显式 verdict 的有状态 workflow。

场景感知物体旋转案例在原始动机任务上验证了该框架。Workflow 生成 train-ready source-target pairs，在 R1 中扩展弱角度数据，并通过固定 LoRA 探针和 SpatialEdit-Bench 协议评估结果。证据以有用方式呈现混合结果：语义和视角指标提升，而一致性和分布指标回归。因此，`inspect` verdict 将渲染质量门控识别为直接瓶颈，而不是把该轮简单视为干净成功或死路。

更广泛的结论是，数据集构建应被视为可检查的闭环系统。下一步是用更严格的渲染质量门控强化内循环，然后测试同一 workflow 是否能支持进一步合成扩容，以及未来混合或真实数据构建任务。

## 13. 可复现性与实现细节

该框架的可复现单元是一个数据集构建轮次。一个轮次从数据集目标和物体概念开始，随后产生 prompt、白底物体图、前景 mask、重建 mesh、场景感知渲染视图、review record、train-ready pairs、下游 checkpoint 和 comparison report。这些产物形成相互关联的状态轨迹，使后续检查能够把 benchmark 回归追溯到产生它的样本和构建阶段。

案例数据集使用 source-target pair schema。source image 是 canonical front view，target image 对应请求的水平旋转。每行记录 source image、target image、instruction text、target rotation、object identifier、object name 和 prompt version。Instruction 使用显式视角名称，例如要求模型将某个命名物体从 front view 旋转到 right-side 或 back view。

Split 构建采用 object-disjoint 规则。Baseline 配置使用 35 个训练物体、7 个验证物体和 8 个测试物体，得到 245 个训练 pair、49 个验证 pair 和 56 个测试 pair。R1 增强配置新增 20 个 train-only 物体和 60 个弱角度 pair，同时保持验证和测试物体不变。这种设计阻止新合成物体泄漏到 held-out splits，并使 R1 比较成为有针对性的训练数据干预。

训练探针使用 Qwen-Image-Edit-2511 进行 LoRA fine-tuning，并采用固定比较协议 [9-11]。已记录运行使用 rank 32、学习率 1e-4、30 个 epoch，并使用 epoch 29 checkpoint 进行评测。探针在 SpatialEdit-Bench 上评估 [12]，该 benchmark 包含 61 个物体和八个角度槽位下的 488 个 rotation pair。指标集合同时包含传统图像指标、表征指标、分布质量和 VLM 视角/一致性评分，与第 8 节中的解释保持一致。

当前 Markdown 技术报告包中的 checkpoint 和 artifact 可用性仍不完整。已跟踪对比使用 exp5 baseline 与 v2 R1 LoRA checkpoints，但大体积 checkpoint 文件和完整推理图因体积原因未打包。完整发布前应补充定性网格、pipeline diagram、loss curve、failure-case figure 和 checkpoint MD5 hash。

### Artifact Index

代码、中间产物与评测日志将在论文公开发表时通过匿名仓库释出。关键超参：LoRA rank=32，学习率=1e-4，训练 epoch=29（所有轮次统一）。数据划分：N_train ≈ 305 pairs，N_val ≈ 49，N_test ≈ 56。Checkpoint hash 在公开仓库中提供。

## 14. References

[1] Tim Brooks, Aleksander Holynski, and Alexei A. Efros. 2023. *InstructPix2Pix: Learning to Follow Image Editing Instructions*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2211.09800.

[2] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. 2023. *MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing*. NeurIPS 2023 Datasets and Benchmarks. arXiv:2306.10012.

[3] Yuxuan Zhang, Huan Ling, Jun Gao, Kangxue Yin, Jean-Francois Lafleche, Adela Barriuso, Antonio Torralba, and Sanja Fidler. 2021. *DatasetGAN: Efficient Labeled Data Factory With Minimal Human Effort*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[4] Blender Foundation. *Blender: Free and Open Source 3D Creation Software*. Official project website: https://www.blender.org/.

[5] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. 2023. *Segment Anything*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

[6] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. 2024. *SAM 2: Segment Anything in Images and Videos*. arXiv:2408.00714.

[7] Team Hunyuan3D et al. 2025. *Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material*. arXiv:2506.15442.

[8] Max Ku, Dongfu Jiang, Cong Wei, Xiang Yue, and Wenhu Chen. 2024. *VIEScore: Towards Explainable Metrics for Conditional Image Synthesis Evaluation*. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), Long Papers. DOI:10.18653/v1/2024.acl-long.663.

[9] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685.

[10] Chenfei Wu et al. 2025. *Qwen-Image Technical Report*. arXiv:2508.02324.

[11] Qwen Team. *Qwen-Image-Edit-2511*. Hugging Face model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2511.

[12] Yicheng Xiao, Wenhu Zhang, Lin Song, Yukang Chen, Wenbo Li, Nan Jiang, Tianhe Ren, Haokun Lin, Wei Huang, Haoyang Huang, Xiu Li, Nan Duan, and Xiaojuan Qi. 2026. *SpatialEdit: Benchmarking Fine-Grained Image Spatial Editing*. arXiv:2604.04911.

[13] OmniGen: Unified Image Generation. Shitao Xiao et al. arXiv:2409.11340 (2024).

[14] Zero-1-to-3: Zero-shot One Image to 3D Object. Ruoshi Liu et al. arXiv:2303.11328 (2023).

[15] ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. Jiazheng Xu et al. arXiv:2304.05977 (2023).

[16] STaR: Bootstrapping Reasoning With Reasoning. Eric Zelikman et al. arXiv:2203.14465 (2022).

[17] Self-Rewarding Language Models. Weizhe Yuan et al. arXiv:2401.10020 (2024).