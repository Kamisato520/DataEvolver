# Code Review: Failure-Aware Stage1 Prompt Regeneration

**日期**: 2026-04-04
**Reviewer**: Claude (Sonnet 4.6)
**Verdict**: `NEEDS_FIXES`

---

## Findings（按严重度排序）

### MEDIUM

#### M1. `_load_latest_trace_text` 取的是最早的 trace，不是最新的
- **文件**: `scripts/run_asset_regeneration_queue.py` 第 217-220 行
- **问题**: 函数名是 `_load_latest_trace_text`，但 `sorted()` 后取 `[0]`（字母序最小 = 最早）。如果一个 attempt 有多个 trace 文件（重试/多视角），拿到的是最旧的那个
- **修复**: 改为 `trace_candidates[-1]`

#### M2. `repair_from_attempt_idx=0` 被 Python 的 falsy 吞掉
- **文件**: `scripts/run_asset_regeneration_queue.py` 第 259 行
- **问题**: `int(job.get("repair_from_attempt_idx") or (...))` —— 当值为 `0` 时，`or` 把它当 falsy，走进默认计算。虽然 attempt_00 通常是原始版本不太可能被 repair，但这是一个隐式假设
- **修复**: 改为显式 `None` 检查：
  ```python
  raw = job.get("repair_from_attempt_idx")
  repair_from = int(raw) if raw is not None else (int(job.get("attempt_idx") or 1) - 1)
  ```

#### M3. `stage1_restart_count` 在 promotion 后不重置
- **文件**: `pipeline/asset_lifecycle.py` 第 566-615 行 (`promote_attempt_to_active`)
- **问题**: promotion 重置了 `status`、清了 `deprecation_reason`、更新了 `replacement_attempts_used`，但 `stage1_restart_count` 始终递增不归零。如果一个资产被 promote 后再次 deprecate，stage1 预算已经用光
- **影响**: 对于短期 V1 跑少量对象影响不大，但长期运行会导致所有曾经 restart 过的资产直接跳过 failure-aware repair
- **修复**: 在 `promote_attempt_to_active` 里加 `entry["stage1_restart_count"] = 0`

#### M4. `build_failure_aware_prompt` 的 `scene_profile` 参数是摆设
- **文件**: `pipeline/stage1_text_expansion.py` 第 282 行签名接收 `scene_profile`，但函数体内完全没用它
- **问题**: 硬编码了 `"overcast roadside scene"` 和 `"Avoid toy-like plastic"` 等文本。如果换场景（不再用 4.blend），这些约束就错了
- **修复**: 至少从 `scene_profile` 里读 `scene_description` 来替换硬编码文本；或者删掉参数避免误导

#### M5. `enqueue_stage1_restart` 在入队时就递增 counter，而非成功后
- **文件**: `pipeline/asset_lifecycle.py` 第 693 行
- **问题**: job 入队就扣预算。如果 job 因 GPU 不可用/网络中断等原因从未执行，restart 次数已被消耗
- **影响**: 结合 M3（不重置），一次未执行的 enqueue 就可能永久浪费一个 restart 机会
- **建议**: 考虑在 attempt 真正 start/complete 时再递增；或者至少在 job 被取消/过期时回退 counter

---

### LOW

#### L1. `_extract_bullet_section` 取最后一个匹配的 heading
- **文件**: `scripts/run_asset_regeneration_queue.py` 第 236-239 行
- **问题**: 没有 `break`，如果 trace 文本里出现两次 `"Major issues:"`，会取最后一次。这可能是有意的（取最终结论），但行为不明显
- **建议**: 加注释说明意图，或加 `break` 取第一个

#### L2. `suggested_fixes` 总是空——`_extract_bullet_section` 对 trace 格式不够稳健
- **文件**: `scripts/run_asset_regeneration_queue.py` 第 282 行
- **问题**: 实际 trace 文本中，"Suggested fixes" 部分可能不以 `Suggested fixes:` 作为独立行 heading 出现（VLM 自由文本格式不固定）。brief 中也确认 attempt_07 的 `suggested_fixes` 为空
- **影响**: repair spec 少了一个重要信号源。当前 stage1 的 repair 只能从 `major_issues` + `issue_tags` 推断修复方向，缺少 VLM 已经给出的具体修复建议
- **建议**: 增加 fallback：从 `agg.json` 的 `suggested_actions` 字段提取；或者对 trace 文本做更宽松的模式匹配（如搜索 "suggest"/"recommend"/"should" 关键词附近的句子）

#### L3. `ceramic_mug` 硬编码特例，无扩展机制
- **文件**: `pipeline/stage1_text_expansion.py` 第 301-306 行
- **问题**: 只有 `ceramic_mug` 有概念特定的几何补强。如果后续 `obj_003`（bicycle）或其他对象也有类似问题，需要继续加 `if concept_name == ...` 分支
- **建议**: 改为数据驱动——在 `TEMPLATE_LIBRARY` 里为每个概念加一个 `repair_geometry_hints` 字段

#### L4. `--template-only` 分支和非 `--template-only` 行为相同
- **文件**: `pipeline/stage1_text_expansion.py` 第 564 行
- **问题**: 正常生成路径中，`if args.scene_conditioned` 和 `else` 分支调用了完全相同的函数和参数（复制粘贴 bug）。虽然这不影响 repair 路径，但说明主路径有潜在问题
- **建议**: 修复三元表达式

#### L5. `stage1_restart_idx` 在两种 job kind 里语义不一致
- **文件**: `pipeline/asset_lifecycle.py`
  - `replacement` job（第 532 行）: `stage1_restart_idx = entry["stage1_restart_count"]`（当前值）
  - `stage1_restart` job（第 688 行）: `stage1_restart_idx = next_restart_idx`（+1 后的值）
- **影响**: 下游消费者读 `stage1_restart_idx` 时，不知道是"已完成几次"还是"当前是第几次"
- **建议**: 统一语义并加注释

---

## Plan 符合度检查

| 目标 | 实现 | 状态 |
|------|------|------|
| repair spec 包含 failure summary | 11 个字段全部覆盖 | ✅ |
| 从 trace.json 提取 major_issues | `_extract_bullet_section` 实现 | ✅ |
| 从 trace.json 提取 suggested_fixes | 实现了但实际提取为空 | ⚠️ |
| failure-aware prompt 显式注入约束 | 5 类信号 → focus clauses | ✅ |
| repair_from_attempt_idx 接线 | lifecycle → queue → runner → stage1 | ✅ |
| API 和 local fallback 双路径 | `call_claude_repair_api` + `build_failure_aware_prompt` | ✅ |
| ceramic_mug 几何补强 | handle/rim/wall/base | ✅ |
| attempt_metadata 记录 lineage | 含 job_kind, repair_from, prompt_source | ✅ |
| 远端灰度已跑通 | obj_002 attempt_07/08 | ✅ |
| structure_consistency 有改善 | major_mismatch → minor_mismatch | ✅ |

---

## 必须修的项（MUST FIX）

1. **M1**: `_load_latest_trace_text` 改为取 `[-1]` 而非 `[0]`
2. **M2**: `repair_from_attempt_idx=0` 的 falsy 问题——改为显式 `None` 检查
3. **M3**: `promote_attempt_to_active` 里重置 `stage1_restart_count = 0`

## 建议优化项（NICE TO HAVE）

4. **M4**: `scene_profile` 参数要么用起来要么删掉
5. **M5**: counter 递增时机移到 attempt start 而非 enqueue
6. **L2**: 增加 `suggested_fixes` 的 fallback 提取路径
7. **L3**: ceramic_mug 特例改为数据驱动

---

## 完成度评价

**实现本身已经成立，不是"有关键缺陷"，而是"已经成立但效果不足"。**

- **链路打通**：failure-aware stage1 从"模板回退"升级为"读取失败摘要 → 重写 prompt"，代码链路完整，远端灰度已证明不是死代码
- **方向正确**：`structure_consistency` 从 `major_mismatch` 降为 `minor_mismatch`，说明几何约束注入起了作用
- **效果不足的根因不在代码本身**：`obj_002` 仍然 reject，主要瓶颈是 T2I 模型对精细几何控制（handle 形态、rim 厚度、陶瓷质感）的可控性有限。这不是 prompt 工程能独立解决的
- **代码质量**：3 个 medium bug 需要修（M1/M2/M3），但都不影响核心逻辑正确性——M1/M2 是边界情况，M3 是长期运行才会暴露的问题。当前灰度跑的 obj_002 结果可信

**一句话**：failure-aware stage1 作为机制已落地成功，但要从 reject 拉到 pass，还需要上游（T2I 模型能力/3D 重建质量）配合，不是单靠 prompt 修复能解决的。
