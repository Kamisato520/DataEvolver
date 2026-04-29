# Failure-Aware Stage1 Review Brief

日期：2026-04-04

本文档用于交给 Claude 做 review。Claude 不需要执行远端任务，只需要审查这次实现是否合理、是否符合当前自动补生目标，并对结果给出判断。

## 1. 本轮任务目标

上一轮自动补生已经支持：

- 资产废弃
- replacement queue
- 两次 seed-based replacement
- 两次失败后自动回到 `stage1`

但之前的 `stage1_restart` 本质上仍然是模板回退，不是真正的失败感知 prompt 重生。

本轮目标是把 `stage1_restart` 改成：

- 读取上一轮失败 attempt 的 smoke review 结果
- 自动提取失败信号
- 基于失败信号重写 prompt
- 再进入 `stage2 -> stage3 -> smoke gate`

也就是把“回到 stage1”从模板式回退，升级成 failure-aware prompt regeneration。

## 2. 本轮实际代码改动

### 2.1 `pipeline/stage1_text_expansion.py`

新增了 failure-aware repair 模式。

已做：

- 新增 `--repair-spec-file`
- 新增 `REPAIR_SYSTEM_PROMPT`
- 新增 `call_claude_repair_api(...)`
- 新增本地 fallback 的 `build_failure_aware_prompt(...)`
- 新增 `build_repair_user_prompt(...)`
- 新增一组辅助函数：
  - `get_seed_concept(...)`
  - `_coerce_text(...)`
  - `_coerce_list(...)`
  - `_shorten(...)`
  - `_contains_any(...)`
  - `_merge_template_meta(...)`
  - `_build_repair_focus_clauses(...)`

当前 repair 模式输入包含：

- 当前 object id / concept
- previous prompt
- previous features
- failure summary

当前 failure summary 支持：

- `reason`
- `detected_verdict`
- `asset_viability`
- `hybrid_score`
- `structure_consistency`
- `issue_tags`
- `lighting_diagnosis`
- `abandon_reason`
- `major_issues`
- `suggested_fixes`
- `trace_text_excerpt`

本地 fallback 不再简单重出模板，而是会根据失败信号显式强调：

- geometry / proportion / silhouette
- material / texture / specular realism
- brightness / midtones / highlights
- framing / full visibility / segmentation cleanliness

对 `ceramic_mug` 还加入了额外几何补强逻辑：

- handle
- rim
- wall thickness
- base ring
- everyday proportions

### 2.2 `scripts/run_asset_regeneration_queue.py`

已改成在 `stage1_restart` 时先构建 repair spec，再调用 stage1。

新增：

- `_load_current_prompt_entry(...)`
- `_load_latest_trace_text(...)`
- `_extract_bullet_section(...)`
- `_build_stage1_repair_spec(...)`

实际行为：

- 从 `repair_from_attempt_idx` 指向的失败 attempt 读取：
  - `prompt.json`
  - `smoke_result.json`
  - `smoke/reviews/*_agg.json`
  - `smoke/reviews/*_trace.json`
- 生成 `stage1_repair_spec.json`
- 调用：
  - `pipeline/stage1_text_expansion.py --repair-spec-file ...`

其他接线：

- `stage1_restart` 现在会把 `repair_from_attempt_idx` 写进 `attempt_metadata.json`
- 后续 replacement follow-up 仍然沿用：
  - `prompt_source_attempt_idx`
  - `use_geometry_suffix`

### 2.3 `pipeline/asset_lifecycle.py`

对 `enqueue_stage1_restart(...)` 增加了：

- `repair_from_attempt_idx`

使得 queue job 能明确记录本轮 stage1 repair 是基于哪个失败 attempt 生成的。

## 3. 本地验证

已完成：

- `python -m py_compile`
  - `pipeline/stage1_text_expansion.py`
  - `pipeline/asset_lifecycle.py`
  - `scripts/run_asset_regeneration_queue.py`
- 用本地构造的 `obj_002` failure spec 运行：
  - `stage1_text_expansion.py --repair-spec-file ... --scene-conditioned --template-only`

验证结论：

- 新 prompt 不再是旧模板回退
- 会显式吸收失败信号
- 会把 `major_mismatch / too dark / plastic-like / rim-handle unclear` 这类问题转成 prompt 约束

## 4. 远端真实灰度执行

### 4.1 远端环境说明

真正执行目录：

- `/aaaidata/zhangqisong/data_build`

本轮远端入口情况：

- 原 AIGC SSH 入口 `58.59.115.26:30022` 当前可 TCP 连接，但不返回 SSH banner，无法直接登录
- 同类入口 `10.123.10.221:30022` 也是同样现象
- 我改为通过可用主机 `wwz` 登录，再在其上直接操作 `/aaaidata/zhangqisong/data_build`

实际远端解释器：

- 系统 `python3` 没有 `torch`
- 最终使用：
  - `/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python`

注意：

- `attempt_07` 第一次误用系统 `python3` 跑 queue runner，导致在 `stage2` 因 `ModuleNotFoundError: torch` 失败
- 这不是代码逻辑 bug，而是执行解释器选错
- 我随后修复了 queue / registry 状态，把 `attempt_07` 恢复成可重跑状态，再用正确解释器重跑

### 4.2 同步到远端的文件

- `/aaaidata/zhangqisong/data_build/pipeline/stage1_text_expansion.py`
- `/aaaidata/zhangqisong/data_build/pipeline/asset_lifecycle.py`
- `/aaaidata/zhangqisong/data_build/scripts/run_asset_regeneration_queue.py`

远端 `py_compile` 已通过。

## 5. `obj_002` 灰度结果

### 5.1 灰度前状态

在本轮开始前，`obj_002` 状态是：

- `active_version = attempt_00`
- `status = manual_reprompt_required`
- `stage1_restart_count = 2`
- 历史上已经跑完：
  - `attempt_01`
  - `attempt_02`
  - `attempt_03`
  - `attempt_04`
  - `attempt_05`
  - `attempt_06`

其中：

- `attempt_06` 的 smoke 结果是：
  - `detected_verdict = revise`
  - `hybrid_score = 0.4805`
  - `structure_consistency = major_mismatch`

### 5.2 本轮新增的自动链路

我把 `obj_002` 重新入队：

- `attempt_07`
- `job_kind = stage1_restart`
- `repair_from_attempt_idx = 6`
- `stage1_restart_idx = 3`

也就是说：

- `attempt_07` 是第一轮真正使用 failure-aware stage1 的实验
- 它不是模板回退，而是显式读取 `attempt_06` 的失败摘要生成新 prompt

### 5.3 `attempt_07` 结果

关键文件：

- `/aaaidata/zhangqisong/data_build/pipeline/data/asset_versions/obj_002/attempt_07/stage1_repair_spec.json`
- `/aaaidata/zhangqisong/data_build/pipeline/data/asset_versions/obj_002/attempt_07/attempt_metadata.json`
- `/aaaidata/zhangqisong/data_build/pipeline/data/asset_versions/obj_002/attempt_07/smoke_result.json`

`attempt_07` 的 repair spec 里明确吸收了这些失败信号：

- `structure_consistency = major_mismatch`
- `issue_tags = [object_too_large, flat_lighting, weak_subject_separation]`
- `major_issues` 中明确提到：
  - scale too large
  - ceramic texture missing / too plastic-like
  - grounding weak
  - lighting too dark / muddy

`attempt_07` 生成的新 prompt 已明显变化，不再是旧模板：

- 强调 complete rounded handle
- 强调 clearly open thick rim
- 强调 believable wall thickness / base ring
- 强调 geometry / material / midtones / highlights

`attempt_07` smoke 结果：

- `detected_verdict = reject`
- `hybrid_score = 0.4703`
- `structure_consistency = minor_mismatch`
- `asset_viability = continue`
- `issue_tags = [floating_visible, object_too_large, flat_lighting]`

最关键的变化：

- 从 `attempt_06` 的 `major_mismatch`
- 下降到 `attempt_07` 的 `minor_mismatch`

这说明：

- failure-aware stage1 至少在几何一致性上起了作用
- 它不是“没有生效”
- 但它还不足以把该资产推过 smoke gate

### 5.4 `attempt_08` 结果

按现有生命周期逻辑，`attempt_07` 失败后自动推进到：

- `attempt_08`
- `job_kind = replacement`
- `prompt_source_attempt_idx = 7`
- `use_geometry_suffix = true`

也就是：

- 先用 failure-aware stage1 产出新 prompt
- 再跑一次 geometry-suffix follow-up

`attempt_08` smoke 结果：

- `detected_verdict = reject`
- `hybrid_score = 0.4658`
- `structure_consistency = minor_mismatch`
- `asset_viability = continue`
- `issue_tags = [floating_visible, object_too_large, underexposed]`

结论：

- `attempt_08` 没有继续恶化到 `major_mismatch`
- 但整体质量依然没有通过 smoke gate
- 主要问题仍集中在：
  - scale too large
  - grounding weak / floating
  - underexposed / flat

### 5.5 当前最终状态

当前远端 registry：

- `active_version = attempt_00`
- `status = manual_reprompt_required`
- `replacement_attempts_used = 8`
- `stage1_restart_count = 3`
- queue 中没有 active job

也就是说：

- 这轮 failure-aware stage1 改动已经真正跑到远端
- 机制有效
- 但 `obj_002` 这个对象仍未被自动救活

## 6. 我对结果的判断

### 6.1 已确认有效的部分

- `stage1_restart` 已从模板回退升级为 failure-aware prompt regeneration
- 远端真实运行已证明这条链路不是死代码
- `attempt_07` 的 prompt 确实读取了失败摘要并改变了生成方向
- `structure_consistency` 从 `major_mismatch` 降为 `minor_mismatch`，说明几何方向被拉正

### 6.2 当前仍然存在的问题

- 虽然 geometry 方向改善了，但最终 reviewer 仍然持续给出 `reject`
- `obj_002` 的主要失败点从“几何完全不对”变成了：
  - 还是太大
  - 还是偏暗
  - grounding 还是弱
  - 材质仍不够陶瓷
- 这意味着：
  - 当前 repair prompt 强度还不够
  - 或者 prompt 改写方向和 T2I/3D 的真实可控性之间仍有偏差

## 7. 希望 Claude 重点 review 的问题

请重点 review 以下点。

### 7.1 代码正确性

请检查：

- `stage1_text_expansion.py` 的 repair 模式是否设计合理
- `run_asset_regeneration_queue.py` 的 repair spec 构建逻辑是否有遗漏
- `repair_from_attempt_idx` 的生命周期接线是否正确
- 是否存在把旧 prompt / 新 prompt / geometry suffix 混用错位的风险

### 7.2 失败摘要抽取是否足够可靠

当前一个明显现象是：

- `major_issues` 在 `attempt_07` 的 `stage1_repair_spec.json` 里提取到了
- 但 `suggested_fixes` 是空的

请检查：

- `_extract_bullet_section(...)` 是否对当前 trace 结构不够稳健
- 是否应该优先从别的字段抽取 `Suggested fixes`
- 是否应该补充从 `issue_tags` / `suggested_actions` 派生结构化 repair guidance

### 7.3 repair prompt 的强度是否足够

请判断：

- 当前 failure-aware prompt 是否还是过于“保守”
- 是否应该更强地注入：
  - realistic product scale
  - clearer ceramic speckle / glaze texture
  - stronger contact realism
  - brighter readable blue glaze
- 是否应该加入更明确的 negative constraints

### 7.4 生命周期设计是否合理

请判断：

- 把 `attempt_07` / `attempt_08` 继续计入 `replacement_attempts_used` 是否合理
- `stage1_restart_count = 3` 后直接进入 `manual_reprompt_required` 是否仍然合适
- 还是应该把 failure-aware stage1 视为一个新 cycle，不与之前模板时代的 cycle 完全等价

## 8. 希望 Claude 的输出格式

请按 code review 方式输出：

1. Findings first
- 按严重度排序
- 给出具体文件和行号
- 重点指出 bug、行为偏差、脆弱实现、或与目标不一致的地方

2. 然后给出裁决
- `APPROVED`
- 或 `NEEDS_FIXES`

3. 最后给出判断
- 这次 failure-aware stage1 是不是“已经成立但效果不足”
- 还是“实现本身仍然存在关键缺陷”

## 9. 总结

一句话总结这轮工作：

> failure-aware stage1 已经真正落地并跑通远端自动补生链路，且对 `obj_002` 的几何一致性产生了正向影响；但它还没有足够强，尚不足以把 `obj_002` 从 `reject` 拉到 smoke pass。
