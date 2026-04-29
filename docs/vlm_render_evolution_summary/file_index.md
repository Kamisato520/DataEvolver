# 文件索引：VLM-Feedback 渲染进化系统（v2 版）

> 本索引记录 v2→v6b 所有关键文件的位置（服务器端 + 本地归档）

---

## 本地归档目录

```
ARIS/docs/vlm_render_evolution_summary/
├── SUMMARY.md                              # v2→v6b 完整进展总结（v2 修订版）
├── SESSION_BRIEF.md                        # 新 session 快速接手 briefing（v2 新增）
├── WORK_REPORT.md                          # 阶段工作汇报/正式总结（v2 新增）
├── experiment_results.json                 # 所有版本汇总数据（v1 遗留，见下注）
├── NEXT_STEPS.md                           # 下一阶段研究方向（v2 修订版）
├── file_index.md                           # 本文件（v2 修订版）
└── results/                                # 结构化实验数据（v2 新增）
    ├── v2_baseline_summary.json            # v2 baseline 10对象分数
    ├── v6_initial_vs_rerender.json         # v6 初始 run vs v6_rerender 实验对比
    ├── v6b_smoke_results.json              # v6b smoke test 3对象结果
    ├── v6b_full_validation_results.json    # v6b full run 10对象完整结果（最权威）
    └── proven_vs_hypothesis.json           # 已证明结论 vs 当前假设 结构化分析
```

> **关于 `experiment_results.json`**：这是 v1 生成的汇总文件，包含跨版本数据，但部分表述存在 v1 已知问题（如错误的版本演化链描述）。建议优先参考 `results/` 下的各个专项 JSON 文件。

---

## results/ 各文件说明

| 文件 | 数据来源 | 用途 |
|------|---------|------|
| `v2_baseline_summary.json` | 服务器文件 `evolution_v2/evolution_summary.json` | v2 baseline 参考分数；含 avg/accepted 统计 |
| `v6_initial_vs_rerender.json` | 服务器文件 `evolution_v6_full/` + `evolution_v6_rerender/` | 解释 v6b Fix A/B/C 的实验依据；flat_no_rim 和 underexposed action 效果对比 |
| `v6b_smoke_results.json` | 服务器文件 `evolution_v6b_smoke/obj_*/evolution_result.json` | v6b pre-full smoke 验证；bounded search 正确性确认 |
| `v6b_full_validation_results.json` | 服务器文件 `evolution_v6b_full/obj_*/evolution_result.json` | **最终权威结果**；10对象完整数据；含 4 项验收标准 |
| `proven_vs_hypothesis.json` | 跨服务器文件分析 | 已证明结论（P1-P7）vs 假设（H1-H4）vs 已证伪（D1-D3）|

---

## 服务器端代码文件（wwz: /aaaidata/zhangqisong/data_build/）

### 核心逻辑

| 文件 | 说明 | 最后修改版本 |
|------|------|-------------|
| `run_evolution_loop.py` | 主控进化循环（v6 完整实现）| v6 |
| `pipeline/stage5_5_vlm_review.py` | VLM 4 维诊断审查（~748 行）| v6 |
| `pipeline/stage5_6_feedback_apply.py` | preset_mode 动作选择（~520 行）| v6 |

### 配置文件

| 文件 | 说明 | 最后修改版本 |
|------|------|-------------|
| `configs/action_space.json` | 动作空间定义（v6b Fix A/B/C 已应用）| v6b |
| `configs/vlm_review_schema.json` | 4 维诊断 JSON schema | v6 |
| `configs/dataset_profiles/rotation_v6.json` | v6 profile（阈值配置，见下）| v6 |
| `configs/dataset_profiles/rotation_v3.json` | 旧版 profile（v2-v5 使用）| v5 |

### rotation_v6.json 关键阈值

```json
{
  "accept_threshold": 0.80,
  "preserve_score_threshold": 0.77,
  "explore_threshold": 0.68,
  "reject_threshold": 0.40,
  "stability_threshold": 0.03,
  "unstable_variance_limit": 0.05,
  "mid_budget": 1,
  "low_budget": 2,
  "improve_eps": 0.01,
  "reference_images_dir": "pipeline/data/images"
}
```

---

## 服务器端结果文件

### 各版本目录

| 目录 | 类型 | 关键文件 |
|------|------|---------|
| `evolution_v2/` | full run（10 obj）| `evolution_summary.json`（含 state_log）|
| `evolution_v4b_full/` | full run（10 obj）| `_partial_cuda{0,1,2}.json` |
| `evolution_v5/` | full run（10 obj）| `_partial_cuda{0,1,2}.json` |
| `evolution_v5b_smoke/` | smoke（3 obj）| `_partial_cuda0.json` |
| `evolution_v6_full/` | full run（10 obj）| `obj_*/evolution_result.json` |
| `evolution_v6_rerender/` | full run（10 obj）| `obj_*/evolution_result.json`（v6b Fix 依据）|
| `evolution_v6b_smoke/` | smoke（3 obj：obj_004/007/008）| `obj_*/evolution_result.json` |
| `evolution_v6b_full/` | full run（10 obj）**最终权威** | `obj_*/evolution_result.json` + `log_gpu*.txt` |

### 参考资产

```
pipeline/data/
├── images/          # Stage 2 T2I 参考图（{obj_id}.png，1328×1328 RGB）
├── meshes/          # Stage 3 生成 mesh（处理后）
├── meshes_raw/      # Stage 3 原始 mesh
├── renders/         # Stage 4 baseline 渲染（默认参数）
└── mesh_qc/         # Mesh 质量检查报告
```

---

## 运行命令参考

### 查看单对象结果

```bash
ssh wwz
cat /aaaidata/zhangqisong/data_build/pipeline/data/evolution_v6b_full/obj_002/evolution_result.json \
  | /home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -m json.tool
```

### 汇总 v6b_full 所有对象分数

```bash
ssh wwz
cd /aaaidata/zhangqisong/data_build
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c "
import json, glob
results = {}
for f in sorted(glob.glob('pipeline/data/evolution_v6b_full/obj_*/evolution_result.json')):
    d = json.load(open(f))
    results[d['obj_id']] = {
        'confirmed': round(d['confirmed_score'], 4),
        'final': round(d['final_hybrid'], 4),
        'delta': round(d['final_hybrid'] - d['confirmed_score'], 4),
        'exit': d['exit_reason']
    }
for k, v in sorted(results.items()):
    print(f\"{k}: confirmed={v['confirmed']}, final={v['final']}, delta={v['delta']}, exit={v['exit']}\")
print(f\"avg_all={sum(v['final'] for v in results.values())/len(results):.4f}\")
"
```

### 验证单元测试（v6b action space）

```bash
ssh wwz
cd /aaaidata/zhangqisong/data_build
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c "
from pipeline.stage5_6_feedback_apply import select_action, load_action_space
aspace = load_action_space()
cs = {'lighting': {'rim_scale': 1.0, 'fill_scale': 1.0}}

r1 = {'lighting_diagnosis': 'flat_no_rim', 'issue_tags': []}
a1 = select_action(r1, 'lighting', cs, {}, aspace, preset_mode=True)
assert a1 == 'L_FILL_DOWN', f'FAIL Fix A: {a1}'  # v6b Fix A

r2 = {'lighting_diagnosis': 'underexposed_global', 'issue_tags': []}
a2 = select_action(r2, 'lighting', cs, {}, aspace, preset_mode=True)
assert a2 == 'L_WORLD_EV_UP', f'FAIL Fix B: {a2}'  # v6b Fix B

print('Unit tests PASS')
"
```

---

## 版本说明

- **v1 文件（SUMMARY/NEXT_STEPS/file_index/experiment_results.json）**：2026-03-31 初稿，基于对话记录整理，包含若干已知错误（版本演化链遗漏 v5b、B 类对象计数错误、结论过度确定等）
- **v2 文件（当前）**：2026-03-31 修订版，修正上述错误，补充 results/ 目录，区分已证明结论与假设
