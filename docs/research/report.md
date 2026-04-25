# 技术报告方向总结：基于 Scene-Aware 渲染与 Free-Form VLM Feedback 的旋转编辑数据集

## 1. 当前判断：现在就可以开始写技术报告

我的核心判断是：**你现在已经可以开始按“技术报告（technical report）”的形式组织这项工作，不需要等到整个 `/data-build` 终局全部完成。**

原因不是因为项目已经“全都做好了”，而是因为你已经具备了一条足够完整、足够清晰、也足够有新意的证据链：

- 有明确的问题定义：自动构建可用于图像编辑训练的受控旋转数据；
- 有清晰的方法主线：scene-aware Blender 渲染 + Qwen3.5 free-form VLM review + AI 读取自由文本后继续调整渲染；
- 有可实例化的数据任务：固定 front view 到其余 7 个目标方位角；
- 有自然的下游验证方式：LoRA 微调是否能提升目标方向的编辑能力。

所以当前最合适的策略不是“等所有事情都做完再写”，而是：

> **先冻结一个可以写成技术报告的 paper version，再围绕这个版本补最必要的证据。**

------

## 2. 这篇报告最适合的定位

这篇文章最好的写法，不是：

> “我们做了一个旋转数据集，然后 LoRA 效果变好了。”

而是：

> **我们提出了一条基于 free-form VLM feedback 的 agentic synthetic data construction 路线，并以 rotation-conditioned image editing dataset 作为第一个具体实例，验证其可以产出训练就绪数据，并对 LoRA 微调有实际帮助。**

这个定位有几个好处：

1. **和当前真正跑通的主线一致**
   当前主线不是单纯 LoRA 项目，而是 Scene-Aware Blender Render + Qwen3.5 VLM Loop。
2. **和终极目标一致**
   终极目标本来就是 `/data-build <描述>` 自动产出训练就绪数据集。当前 rotation 数据集正好可以作为这一目标的第一个落地实例。
3. **能把“小规模数据集”讲成“聚焦验证场景”**
   50 个对象不算大，但如果你把它定义成一个清晰、可控、可评估的 testbed，它就不是“小”，而是“聚焦”。

------

## 3. 关于数据集设计：当前方案是对的，而且很适合第一版

### 3.1 基本数据定义

你目前的设计是：

- 约 50 个 object；
- 每个 object 渲染 8 个水平视角；
- 固定 `0° = front view` 作为输入；
- 目标是其余 7 个视角；
- 所以每个 object 有 7 对训练对；
- 总共约 `50 × 7 = 350` 对训练样本。

这是一个非常适合技术报告首发版的规模。

### 3.2 为什么这个设计好

这个数据设计有几个明显优点：

#### （1）任务定义极其清楚

输入永远是 front view，输出永远是某个目标视角。训练目标非常明确，没有开放式描述任务常见的语义歧义。

#### （2）prompt 简洁、统一、可控

你打算使用自然语言视角短语，而不是角度数值，例如：

- front view
- front-right quarter view
- right side view
- back-right quarter view
- back view
- back-left quarter view
- left side view
- front-left quarter view

这比“rotate by 90 degrees”更像真实用户会说的话，也更像图像编辑指令的自然语言形式。

#### （3）评估非常容易做

因为目标方位离散而明确，所以视角正确性、身份保持、编辑自然度等维度都很容易设计评估方案。

#### （4）和当前 scene-aware + VLM loop 主线天然兼容

rotation 本来就是当前 active 主线里已经在推进的任务方向之一，所以不是另起炉灶。

------

## 4. 关于 prompt 设计：建议继续用英文短语，但尽量更统一、更短

你当前这套 prompt 设计总体是合理的：

- 0°：front view
- 45°：front-right quarter view
- 90°：right side view
- 135°：back-right quarter view
- 180°：back view
- 225°：back-left quarter view
- 270°：left side view
- 315°：front-left quarter view

我总体支持这个方向，但建议考虑一个更统一、略微更简洁的版本，例如：

- front view
- front-right view
- right side view
- back-right view
- back view
- back-left view
- left side view
- front-left view

然后统一使用同一个模板：

```text
Rotate this object from front view to {target_view}.
```

这样做的好处是：

- 模板固定，语言噪声更低；
- 更像一个受控的 instruction-editing task；
- 训练和推理可以保持严格一致；
- 写技术报告时更工整。

如果你希望保留“quarter view”这种更细致的术语，也不是不行，但我个人更偏向短一点、稳一点的短语集合。

------

## 5. 350 对是否足够：对技术报告第一版来说，够了

如果目标是一个**先挂 arXiv 的技术报告版本**，350 对并不算寒酸，关键在于你怎么组织叙事。

你完全可以把数据规模定义清楚：

- **基础视角图像总数**：50 objects × 8 views = 400 rendered images；
- **图像编辑训练对总数**：50 objects × 7 target views = 350 editing pairs。

这两个量需要在文中明确区分：

- `rendered view set`
- `editing pair set`

这样写会显得更专业，也更便于别人理解你的数据是如何从多视角渲染资产转成训练对的。

------

## 6. 一个非常重要的实验原则：train / val / test 必须按 object 拆分

这是我最强烈建议你提前定死的地方之一。

### 一定不要按 pair 随机拆分

因为同一个物体的不同视角高度相关，如果按 pair 随机划分，很容易出现：

- 训练集见过同一个物体的别的角度；
- 测试集只是同一物体的另一个角度；
- 结果看起来很好，但泛化其实很虚。

### 更合理的做法

建议你按 **object-disjoint split** 切分：

例如：

- train：35 个 object
- val：5 个 object
- test：10 个 object

或者：

- train：40 个 object
- val：5 个 object
- test：5 个 object

只要原则是**同一个 object 不能同时出现在训练和测试**，这篇技术报告的可信度就会高很多。

------

## 7. 这篇技术报告最核心的价值，不是“LoRA 有提升”，而是“数据构建方法本身成立”

LoRA 提升很重要，但**它不能成为整篇技术报告唯一的支撑点**。

因为别人很容易反问：

- 是不是任何 synthetic data 都能让 base model 变好？
- 是不是因为做了 LoRA，所以本来就会比 base 好？
- 是不是 improvement 只在你自己的 judge 上成立？

所以更稳的证据结构应该有三层：

### 7.1 数据构建层

证明你的 pipeline 不是随便渲染几张图，而是通过 scene-aware + free-form VLM loop 把数据质量逐轮推高。

### 7.2 数据质量层

证明你最终筛出来的数据在视觉质量、视角正确性、材质和 grounding 等方面是可接受的。

### 7.3 下游效用层

证明这些数据不仅“看起来不错”，而且真的能用于训练，并带来实际的编辑能力提升。

也就是说：

> **LoRA 是必要的 downstream validation，但不是整篇报告的唯一核心。**

------

## 8. 技术报告最自然的主叙事

我建议你把整篇报告的主叙事定成这样：

> **我们正在构建一个自动化 synthetic data system；本文报告其中一个已经形成闭环的子任务：以 canonical front view 为输入，自动构建面向受控视角编辑的 rotation dataset，并验证其训练价值。**

这个叙事特别好，因为它同时完成了三件事：

1. 把你现在做的 rotation 任务放进大系统愿景中；
2. 不会把第一篇文章写得过大过满；
3. 让“技术报告”这种形式显得合理而自然。

换句话说，这篇文章不是说“我们已经完全实现了通用数据工厂”，而是说：

> **我们在通往 training-ready synthetic data system 的路上，已经验证了其中一条关键路径。**

这个口径非常稳。

------

## 9. 建议的文章贡献（contributions）写法

我建议把贡献写得克制，但有力量。

可以组织成三条：

### Contribution 1

提出一个 **free-form VLM-guided synthetic data refinement loop**：
AI 不再只依赖 rigid score 或固定 controller，而是直接读取 reviewer 的自由文本反馈，并决定下一轮渲染动作。

### Contribution 2

将该 loop 落地到 **scene-aware Blender rendering** 中：
系统可围绕 lighting、grounding、shadow、material、color consistency 等问题持续优化渲染结果。

### Contribution 3

以 **view-controlled rotation editing dataset** 为实例：
构建一个 50-object、350-pair 的数据集，并验证它对 LoRA 微调具有实际训练价值。

这三条里，真正体现论文味的，是前两条；第三条负责把方法落地和封口。

------

## 10. 技术报告的推荐结构

下面是一套很适合当前阶段的文章结构。

### 10.1 Introduction

说明：

- 高质量图像编辑训练数据昂贵、难以构建；
- 受控视角编辑尤其缺乏便宜、可控、成对的数据；
- 传统 synthetic pipeline 常常只靠简单评分或 rigid schema，难以针对复杂视觉问题做持续修正；
- 因此需要一种能够读取自由文本视觉反馈、并自动继续优化渲染的 synthetic data construction 方法。

### 10.2 System Overview

介绍整体系统链路：

- object asset preparation；
- scene-aware Blender render；
- free-form VLM review；
- AI 读取 reviewer 文本并决定下一轮动作；
- 直到 reviewer 明确 keep / acceptable 才停止。

### 10.3 Rotation Dataset Construction

详细讲 rotation 数据集：

- 50 个 object；
- 8 个水平视角；
- front view 作为 canonical input；
- 其余 7 个目标视角构成 350 对训练对；
- prompt 模板与 metadata 设计。

### 10.4 Quality Control and Filtering

这是最体现系统性的部分：

- reviewer 看什么；
- trace/free-form text 如何被读取；
- 哪些问题会触发继续 refine；
- keep / acceptable 的判定逻辑；
- 最终数据是如何被筛选出来的。

### 10.5 LoRA Training Setup

交代：

- base model；
- LoRA 设置；
- 输入输出格式；
- source image + instruction → target image 的监督方式；
- train / val / test split。

### 10.6 Evaluation

至少评估：

- 目标视角正确性；
- identity preservation；
- overall edit quality。

### 10.7 Baselines

至少包含：

- base model without LoRA；
- base model + LoRA on your dataset；
- 最好再加：same-size baseline synthetic dataset 上的 LoRA。

### 10.8 Failure Cases and Limitations

这里应该诚实地写：

- 某些 pair 可能长时间卡在 flat lighting / color shift；
- 当前 action space 对某些问题的修复能力有限；
- 小规模对象集的类别覆盖仍然有限；
- 当前只验证了 rotation 一种任务。

这一节对技术报告非常重要，反而会增加可信度。

------

## 11. 关于评价指标：不要只看“图更好看了没有”

你的任务本质上是**受控编辑**，所以评估最好围绕控制目标来设计，而不是泛泛地评整体画质。

我建议至少分成三个维度：

### 11.1 View Accuracy

模型是否真的把 front view 变成了 prompt 指定的目标方位角。

### 11.2 Object Consistency

生成结果是否仍然保持同一个 object 的主结构、形状特征和关键外观属性。

### 11.3 Overall Edit Quality

结果图像是否自然、是否有明显伪影、遮挡错误、结构断裂、材质怪异等问题。

如果条件允许，还可以加：

### 11.4 Prompt Alignment

模型是否准确理解了 target view phrase，而没有偏到相邻方位或错误方向。

这些维度哪怕用 VLM-based judge 来做，也会比只报一个总分更合理。

------

## 12. baseline 应该怎么设，才能证明“是你的数据构建方法有用”

如果只比较：

- base model
- base + LoRA on your dataset

那只能证明“微调有帮助”，还不能强力证明“你的数据构建策略特别有效”。

所以我建议至少补一个 **equal-size baseline dataset**：

### 可选 baseline 方案

#### 方案 A：不做 scene-aware refinement 的简单渲染数据

也就是：

- 同样是 50 个 object；
- 同样是 8 个视角；
- 但不经过 free-form VLM loop 精修；
- 直接拿初始渲染结果构建训练对。

#### 方案 B：不使用 free-form feedback，只用简单 score / rigid controller 的数据

这会更能对应你的系统贡献点。

#### 方案 C：同样规模的人工或常规 synthetic 数据集

如果你能拿到合适对照，也可以做，但第一版不一定必须。

只要你能证明：

> **同样的数据量下，你这条 scene-aware + free-form VLM-guided 路线产出的数据，训练效果更好或至少不差。**

那整篇报告就会非常有说服力。

------

## 13. 这篇技术报告最适合强调的核心结论

我建议最终结论不要写成：

> “我们构建了一个 50-object 的旋转数据集。”

而要写成：

> **我们证明了：即使在较小规模上，基于 scene-aware rendering 与 free-form VLM refinement 自动构建的 rotation editing data，也足以作为训练就绪数据，为受控视角编辑带来可测的能力提升。**

这句话的好处在于：

- 不把 claim 吹太大；
- 但明确指出“训练就绪”这个核心价值；
- 也把“自动构建的数据是有用的”这件事讲清楚了。

------

## 14. 当前最值得避免的两个问题

### 14.1 不要把 LoRA 结果写成整篇报告唯一的价值来源

LoRA 只是 downstream proof，不是整篇文章全部的灵魂。

真正的灵魂是：

- synthetic data 是怎么被自动构建出来的；
- VLM free-form feedback 是怎么被接入 decision loop 的；
- scene-aware 渲染是怎么被持续改进的；
- 为什么最终筛出来的数据值得训练。

### 14.2 不要把文章写成“大而全的通用系统已经彻底完成”

第一版技术报告更适合诚实、克制的口径：

- 我们验证了一条关键路径；
- 我们展示了其可行性；
- 我们分析了当前失败模式；
- 我们指出后续扩展方向。

这比“全都解决了”的语气更可信，也更适合现在这个阶段。

------

## 15. 对题目方向的建议

建议题目采用：

- **主标题讲系统/目标**
- **副标题讲 rotation 是实例**

例如：

### 偏系统型

**Toward Training-Ready Synthetic Data for View-Controlled Image Editing**
*A Technical Report on Scene-Aware Rendering, Free-Form VLM Feedback, and a 50-Object Rotation Dataset*

### 偏任务型

**Scene-Aware Synthetic Rotation Data for Image Editing**
*A Technical Report*

### 偏方法型

**Free-Form VLM-Guided Synthetic Data Construction for View-Controlled Image Editing**

整体上，我更推荐第一类写法，因为它能把：

- training-ready synthetic data
- free-form VLM loop
- rotation dataset 实例

这三层统一起来。

------

## 16. 最终建议：这件事现在最好的写法是什么

如果要把我前面的所有意见浓缩成一句话，那就是：

> **这篇技术报告最好的写法，不是“我们做了一个旋转 LoRA 数据集”，而是“我们做了一个 free-form VLM 驱动的 scene-aware synthetic data construction 系统，rotation 数据集是它的第一个验证场景，而 LoRA 结果证明这批自动构建的数据确实对训练有用”。**

这个表述最符合你现在的真实进展，也最适合先挂 arXiv 的技术报告形态。

它既不虚，也不小气；既能立住方法，也能落到可测的下游结果；既承接当前主线，也为之后扩展到更大的 `/data-build` 愿景留了空间。

------

## 17. 一句话结论

**结论：你的 50 object / 350 pair rotation 方案，已经足够作为一篇技术报告第一版的核心实验实例；真正应该强调的是“自动构建训练就绪数据的方法成立”，而不是仅仅“LoRA 变好了”。**

这份内容是基于你当前 session handoff 和更新后的 CLAUDE.md 主线整理的。