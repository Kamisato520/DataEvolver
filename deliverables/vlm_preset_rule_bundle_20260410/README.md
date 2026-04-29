# Scene VLM 预设规则示例包

这个文件夹整理的是“我亲自接手难例之前”的那一套预设规则链路，也就是：

1. 先定义 VLM 要看哪些维度
2. 再约束 VLM 输出哪些结构化字段
3. 再把这些输出映射成离散动作
4. 最后由 monitor 自动决定下一轮怎么改

这套逻辑主要用于 scene insertion / scene-aware render loop。

## 文件列表

- `configs/vlm_review_schema.json`
  - 定义 VLM 结构化输出 schema
  - 重点看评分维度、issue tags、suggested actions、诊断字段

- `pipeline/stage5_5_vlm_review.py`
  - 真正发给 VLM 的 prompt 和结果聚合逻辑
  - 会把多视角结果聚成 `agg.json`

- `configs/scene_action_space.json`
  - 定义可执行的离散动作空间
  - 定义从 issue/diagnosis 到 action 的预设映射

- `scripts/run_scene_agent_monitor.py`
  - 自动 monitor 主控脚本
  - 读取 `agg.json + trace.json`，决定下一轮动作

- `scripts/run_scene_agent_step.py`
  - 单轮 step runner
  - 负责落盘 round 结果和 review 输出路径

## 建议阅读顺序

建议按下面顺序看：

1. `configs/vlm_review_schema.json`
2. `pipeline/stage5_5_vlm_review.py`
3. `configs/scene_action_space.json`
4. `scripts/run_scene_agent_monitor.py`
5. `scripts/run_scene_agent_step.py`

如果只想最快理解“预设维度”和“预设动作”：

- 先看 `vlm_review_schema.json`
- 再看 `scene_action_space.json`
- 最后看 `run_scene_agent_monitor.py` 里的 `decide_actions()`

## 预设的评价维度是怎么做的

在 `configs/vlm_review_schema.json` 里，主分数字段是这 5 个：

- `lighting`
- `object_integrity`
- `composition`
- `render_quality_semantic`
- `overall`

可以把它理解成：

- `lighting`
  - 看亮度、阴影、光线匹配、整体照明自然不自然

- `object_integrity`
  - 看物体是否完整、有没有明显畸形、断裂、低模崩坏、材质身份错误

- `composition`
  - 看物体大小、位置、接地、和背景关系、是否 pasted-on

- `render_quality_semantic`
  - 看“渲染出来的东西是否像它应该是的那个物体”
  - 这里既看画质，也看语义身份

- `overall`
  - 总体印象分

除了这 5 个主分数，schema 还要求 VLM 输出辅助结构化字段：

- `issue_tags`
- `suggested_actions`
- `lighting_diagnosis`
- `structure_consistency`
- `color_consistency`
- `physics_consistency`
- `asset_viability`

这些字段的作用不是“多打几项分”，而是给后面的 monitor 提供可执行的决策信号。

## VLM prompt 是怎么组织的

在 `pipeline/stage5_5_vlm_review.py` 里，scene 模式会使用 `scene_insert` 这一套 review 流程。

这部分做了几件事：

1. 选 representative views
2. 让 VLM 对每个视角输出结构化 JSON
3. 把多视角结果聚合成一个 `agg.json`

你重点可以看这些位置：

- `ISSUE_TAGS`
- `LIGHTING_DIAGNOSIS_VALUES`
- `review_mode == "scene_insert"`
- `agg_vlm_scores`

这个阶段的核心输出包括：

- 聚合后的五个主分数
- 最常见的 `issue_tags`
- 最常见的 `suggested_actions`
- 聚合后的 `lighting_diagnosis`

也就是说，VLM 在这里并不只是“打一段自然语言评语”，而是被要求输出一个后续脚本可以直接消费的结构化结果。

## 预设动作空间是怎么做的

在 `configs/scene_action_space.json` 里，预设动作是离散的，而不是无限自由修改。

例如里面有这些典型动作：

- `O_SCALE_UP_10`
- `O_SCALE_DOWN_10`
- `O_LOWER_SMALL`
- `ENV_STRENGTH_UP`
- `ENV_STRENGTH_DOWN`
- `S_CONTACT_SHADOW_UP`
- `M_VALUE_UP`
- `M_VALUE_DOWN`
- `M_HUE_WARM_NEG`
- `M_SATURATION_DOWN`

这些动作的设计思路是：

- 物体位置类
- 物体尺度类
- 环境光类
- 接地阴影类
- 材质 HSV / roughness / specular 类

然后这个文件还定义了“问题 -> 动作”的映射，比如：

- `underexposed`
  - 倾向映射到 `ENV_STRENGTH_UP`

- `flat_lighting`
  - 倾向映射到环境旋转或 key light 调整

- `color_shift`
  - 倾向映射到 `M_VALUE_DOWN_STRONG / M_HUE_WARM_NEG / M_SATURATION_DOWN`

- `floating_visible`
  - 倾向映射到 `O_LOWER_SMALL`

- `scale_implausible`
  - 倾向映射到 `O_SCALE_UP_10 / O_SCALE_DOWN_10`

所以在“预设规则阶段”，系统并不会直接生成任意 Blender 改动，而是只能在一个受控动作空间里选动作。

## monitor 是怎么把 VLM 输出变成下一轮动作的

`scripts/run_scene_agent_monitor.py` 是这一套预设规则真正的控制器。

关键函数是：

- `_suggested_actions()`
- `_prune_conflicting_actions()`
- `decide_actions()`

它的逻辑大致是：

1. 读取 `agg.json`
2. 读取 `trace.json` 或 reviewer 自由文本
3. 提取：
   - `issue_tags`
   - `suggested_actions`
   - `trace_text`
4. 合并显式建议和文本启发式规则
5. 裁掉互相冲突的动作
6. 输出下一轮 actions

这一层还有一个硬限制：

- `HARD_MAX_ROUNDS = 15`

也就是说，预设 monitor 并不是无限试，而是一个带上限的自动规则回路。

## 为什么后面会转成我亲自接手

这套预设规则的优点是：

- 吞吐高
- 可以先批量清掉常见问题
- 适合大多数标准 case

但它的局限也很明确：

- 只能在离散动作空间里改
- 改动粒度有限
- 有些 case 会出现“参数改了，但画面几乎没变”
- 有些 case 的问题不是简单亮一点、暗一点、缩一点就能解决
- 对“材质身份错误 / 白模无颜色 / 资产级问题”不够强

所以后面我才会从“预设规则阶段”切换到“我自己直接接手难例”，也就是：

- 直接读 VLM 的自由文本
- 不只依赖离散动作
- 必要时直接改 control state
- 必要时直接改渲染逻辑或材质适配代码

## 一句话总结

这套预设配置的本质是：

**先用固定 schema 让 VLM 给出结构化评价，再用固定动作空间和 monitor 规则把这些评价转成下一轮渲染动作。**

如果你要快速复核我当时的设计，优先看：

1. `configs/vlm_review_schema.json`
2. `configs/scene_action_space.json`
3. `scripts/run_scene_agent_monitor.py`
