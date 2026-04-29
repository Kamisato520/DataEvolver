# FIGURE INDEX — VLM-Feedback 渲染进化系统

> **拉取时间**：2026-03-31
> **服务器**：wwz (`/aaaidata/zhangqisong/data_build/`)
> **本地根目录**：`figures/raw/`
> **图片选角**：统一使用 `az000_el+00`（正面视角，0° 方位角，0° 仰角）

---

## 目录结构总览

```
figures/raw/
├── obj_002/          # 成功修复案例（underexposed_global → +0.027）
│   ├── reference/    # Stage 2 T2I 参考图
│   ├── baseline/     # Stage 4 默认参数渲染
│   ├── v6_rerender/  # v6 rerender run 中的 action attempts
│   ├── v6b_full/     # v6b full run 最终结果
│   └── json/         # evolution_result.json（v6_rerender + v6b_full）
├── obj_009/          # 高分无需优化（accepted_baseline，0.8053）
│   ├── reference/
│   ├── baseline/
│   └── json/
├── obj_004/          # flat_no_rim 安全退出（low_exhausted，0.6561）
│   ├── reference/
│   ├── baseline/
│   ├── v6_rerender/  # L_RIM_UP + L_RIM_FILL_UP（v6 旧 action space）
│   ├── v6b_full/     # L_FILL_DOWN + L_KEY_SHADOW_SOFT（v6b Fix A）
│   └── json/
├── obj_007/          # flat_no_rim borderline（low_exhausted，0.65045）
│   ├── reference/
│   ├── baseline/
│   ├── v6_rerender/
│   ├── v6b_full/
│   └── json/
└── obj_008/          # flat_no_rim 稳定失败（low_exhausted，0.6483）
    ├── reference/
    ├── baseline/
    ├── v6_rerender/
    ├── v6b_full/
    └── json/
```

---

## obj_002 — underexposed_global 成功修复案例

**核心结论**：v6b Fix B 生效。L_WORLD_EV_UP 将 confirmed=0.486 → final=0.513（+0.027）。

| 本地路径 | 服务器原始路径 | 实验轮次 | action | 结果 | 选图原因 |
|---------|-------------|---------|--------|------|---------|
| `obj_002/reference/obj_002.png` | `pipeline/data/images/obj_002.png` | — | — | — | T2I 参考图（目标外观） |
| `obj_002/baseline/az000_el+00.png` | `pipeline/data/renders/obj_002/az000_el+00.png` | Stage 4 | — | confirmed=0.486 | 默认参数渲染基准 |
| `obj_002/v6_rerender/L_KEY_UP_attempt_az000_el+00.png` | `evolution_v6_rerender/obj_002/renders_u01/obj_002/az000_el+00.png` | v6_rerender u01 | L_KEY_UP | delta=-0.003（失败）| 展示 Fix B 前：错误 action 无效 |
| `obj_002/v6_rerender/L_WORLD_EV_UP_accepted_az000_el+00.png` | `evolution_v6_rerender/obj_002/renders_u02/obj_002/az000_el+00.png` | v6_rerender u02 | L_WORLD_EV_UP | delta=+0.012（接受）| Fix B 依据：首次证明 L_WORLD_EV_UP 有效 |
| `obj_002/v6b_full/L_WORLD_EV_UP_FINAL_az000_el+00.png` | `evolution_v6b_full/obj_002/renders_u01/obj_002/az000_el+00.png` | v6b_full u01 | L_WORLD_EV_UP | delta=+0.027（接受）| **最终最佳渲染**；Fix B 在 v6b 正式运行中复现 |
| `obj_002/json/evolution_result_v6_rerender.json` | `evolution_v6_rerender/obj_002/evolution_result.json` | v6_rerender | — | — | 完整 state_log |
| `obj_002/json/evolution_result_v6b_full.json` | `evolution_v6b_full/obj_002/evolution_result.json` | v6b_full | — | — | 完整 state_log（权威） |

**注**：obj_002 在 v6b_full 中仅需 1 次 attempt（L_WORLD_EV_UP）即触发 `accepted_after_try`，因此只有 `renders_u01`，无 `renders_u02`。

---

## obj_009 — 高分基准无需优化

**核心结论**：controller 正确识别高质量 render 并安全退出（accepted_baseline）。

| 本地路径 | 服务器原始路径 | 实验轮次 | 结果 | 选图原因 |
|---------|-------------|---------|------|---------|
| `obj_009/reference/obj_009.png` | `pipeline/data/images/obj_009.png` | — | — | T2I 参考图 |
| `obj_009/baseline/az000_el+00.png` | `pipeline/data/renders/obj_009/az000_el+00.png` | Stage 4 | confirmed=0.805 | **即为最终渲染**；baseline 已达 accept_threshold=0.80 |
| `obj_009/json/evolution_result_v6b_full.json` | `evolution_v6b_full/obj_009/evolution_result.json` | v6b_full | 0 attempts | state_log 为空，exit=accepted_baseline |

**缺失说明**：obj_009 在 v6b_full 中无任何 `renders_u0x`（0 次 action，直接接受 baseline），这是符合预期的正常情况，非数据缺失。

---

## obj_004 — flat_no_rim 安全退出（最典型失败案例）

**核心结论**：新旧 action space 均无法突破 flat_no_rim，但 bounded search 保证 final=confirmed，零退化。

| 本地路径 | 服务器原始路径 | 实验轮次 | action | 结果 | 选图原因 |
|---------|-------------|---------|--------|------|---------|
| `obj_004/reference/obj_004.png` | `pipeline/data/images/obj_004.png` | — | — | — | T2I 参考图 |
| `obj_004/baseline/az000_el+00.png` | `pipeline/data/renders/obj_004/az000_el+00.png` | Stage 4 | — | confirmed=0.656 | 基准渲染 |
| `obj_004/v6_rerender/L_RIM_UP_failed_az000_el+00.png` | `evolution_v6_rerender/obj_004/renders_u01/obj_004/az000_el+00.png` | v6_rerender u01 | L_RIM_UP | delta=-0.008（失败）| v6 旧策略：rim 灯无效 |
| `obj_004/v6_rerender/L_RIM_FILL_UP_failed_az000_el+00.png` | `evolution_v6_rerender/obj_004/renders_u02/obj_004/az000_el+00.png` | v6_rerender u02 | L_RIM_FILL_UP | delta=-0.008（失败）| compound action 同样无效 |
| `obj_004/v6b_full/L_FILL_DOWN_failed_az000_el+00.png` | `evolution_v6b_full/obj_004/renders_u01/obj_004/az000_el+00.png` | v6b_full u01 | L_FILL_DOWN | delta=-0.008（失败）| v6b Fix A 新策略：fill down 也无效 |
| `obj_004/v6b_full/L_KEY_SHADOW_SOFT_failed_az000_el+00.png` | `evolution_v6b_full/obj_004/renders_u02/obj_004/az000_el+00.png` | v6b_full u02 | L_KEY_SHADOW_SOFT | delta=-0.008（失败）| 穷举后 low_exhausted，final=confirmed=0.656 |
| `obj_004/json/evolution_result_v6_rerender.json` | `evolution_v6_rerender/obj_004/evolution_result.json` | v6_rerender | — | — | 完整 state_log |
| `obj_004/json/evolution_result_v6b_full.json` | `evolution_v6b_full/obj_004/evolution_result.json` | v6b_full | — | — | 完整 state_log（权威）|

---

## obj_007 — flat_no_rim borderline（最有争议的案例）

**核心结论**：v6_full 中 L_RIM_UP 曾显示 +0.011（被误接受），v6_rerender 重测仅 +0.006（低于 eps=0.01）。v6b Fix A 后同样低于 eps。证明稳定性确认机制的必要性（P7）。

| 本地路径 | 服务器原始路径 | 实验轮次 | action | 结果 | 选图原因 |
|---------|-------------|---------|--------|------|---------|
| `obj_007/reference/obj_007.png` | `pipeline/data/images/obj_007.png` | — | — | — | T2I 参考图 |
| `obj_007/baseline/az000_el+00.png` | `pipeline/data/renders/obj_007/az000_el+00.png` | Stage 4 | — | confirmed=0.650 | 基准渲染 |
| `obj_007/v6_rerender/L_RIM_UP_below_eps_az000_el+00.png` | `evolution_v6_rerender/obj_007/renders_u01/obj_007/az000_el+00.png` | v6_rerender u01 | L_RIM_UP | delta=+0.006（低于 eps，失败）| 与 v6_full 正值矛盾的关键图 |
| `obj_007/v6_rerender/L_RIM_FILL_UP_below_eps_az000_el+00.png` | `evolution_v6_rerender/obj_007/renders_u02/obj_007/az000_el+00.png` | v6_rerender u02 | L_RIM_FILL_UP | delta=+0.0002（失败）| compound 同样几乎无效 |
| `obj_007/v6b_full/L_FILL_DOWN_below_eps_az000_el+00.png` | `evolution_v6b_full/obj_007/renders_u01/obj_007/az000_el+00.png` | v6b_full u01 | L_FILL_DOWN | delta=+0.006（低于 eps，失败）| Fix A 后仍然低于 eps |
| `obj_007/v6b_full/L_KEY_SHADOW_SOFT_below_eps_az000_el+00.png` | `evolution_v6b_full/obj_007/renders_u02/obj_007/az000_el+00.png` | v6b_full u02 | L_KEY_SHADOW_SOFT | delta=+0.0002（失败）| 穷举退出，final=confirmed=0.650 |
| `obj_007/json/evolution_result_v6_rerender.json` | `evolution_v6_rerender/obj_007/evolution_result.json` | v6_rerender | — | — | |
| `obj_007/json/evolution_result_v6b_full.json` | `evolution_v6b_full/obj_007/evolution_result.json` | v6b_full | — | — | |

**注**：v6_full 中 obj_007 的 L_RIM_UP 被单测 accepted（+0.011）但未拉取该图，因为 v6_full 存在 rendering application bug 嫌疑（delta=0.0 问题），其渲染图可信度低。

---

## obj_008 — flat_no_rim 稳定失败（最干净的反例）

**核心结论**：新旧 action space 一致地产生负 delta（-0.009），无任何 borderline 情况。是展示 flat_no_rim 上限的最佳案例。

| 本地路径 | 服务器原始路径 | 实验轮次 | action | 结果 | 选图原因 |
|---------|-------------|---------|--------|------|---------|
| `obj_008/reference/obj_008.png` | `pipeline/data/images/obj_008.png` | — | — | — | T2I 参考图 |
| `obj_008/baseline/az000_el+00.png` | `pipeline/data/renders/obj_008/az000_el+00.png` | Stage 4 | — | confirmed=0.648 | 基准渲染 |
| `obj_008/v6_rerender/L_RIM_UP_failed_az000_el+00.png` | `evolution_v6_rerender/obj_008/renders_u01/obj_008/az000_el+00.png` | v6_rerender u01 | L_RIM_UP | delta=-0.009（失败）| v6 旧策略一致性负反馈 |
| `obj_008/v6_rerender/L_RIM_FILL_UP_failed_az000_el+00.png` | `evolution_v6_rerender/obj_008/renders_u02/obj_008/az000_el+00.png` | v6_rerender u02 | L_RIM_FILL_UP | delta=-0.009（失败）| |
| `obj_008/v6b_full/L_FILL_DOWN_failed_az000_el+00.png` | `evolution_v6b_full/obj_008/renders_u01/obj_008/az000_el+00.png` | v6b_full u01 | L_FILL_DOWN | delta=-0.009（失败）| v6b Fix A 同样负反馈 |
| `obj_008/v6b_full/L_KEY_SHADOW_SOFT_failed_az000_el+00.png` | `evolution_v6b_full/obj_008/renders_u02/obj_008/az000_el+00.png` | v6b_full u02 | L_KEY_SHADOW_SOFT | delta=-0.009（失败）| |
| `obj_008/json/evolution_result_v6_rerender.json` | `evolution_v6_rerender/obj_008/evolution_result.json` | v6_rerender | — | — | |
| `obj_008/json/evolution_result_v6b_full.json` | `evolution_v6b_full/obj_008/evolution_result.json` | v6b_full | — | — | |

---

## 缺失图片说明

| 预期图片 | 查找过的目录 | 缺失原因 |
|---------|-----------|---------|
| obj_009 evolution renders | `evolution_v6b_full/obj_009/` | 正常缺失：obj_009 exit=accepted_baseline，未执行任何 action，无 `renders_u0x` 目录 |
| v6_full (初始 run) 中 obj_004/007/008 的 renders | `evolution_v6_full/obj_*/` | 未拉取：v6_full 存在 rendering application bug（delta=0.0），该 run 的 render 可信度低，不适合汇报展示 |
| obj_002 v6b_full renders_u02 | `evolution_v6b_full/obj_002/` | 正常缺失：obj_002 在 v6b_full 中仅 1 次 attempt 即成功，无第二次 render |

---

## 汇报场景建议

| 汇报主题 | 推荐图片组合 |
|---------|-----------|
| "系统能改善什么" | obj_002：reference + baseline + v6b_full/L_WORLD_EV_UP_FINAL（3 张对比）|
| "系统知道何时不动" | obj_009：reference + baseline（2 张） |
| "flat_no_rim 是当前上限" | obj_004/obj_008：baseline + v6b_full 2 张（展示所有 action 均无效）|
| "v6 vs v6b action space 对比" | obj_004：v6_rerender/L_RIM_UP + v6b_full/L_FILL_DOWN（展示 Fix A 的替换）|
| "borderline VLM 噪声" | obj_007：v6_rerender/L_RIM_UP（+0.006 低于 eps） vs evolution_result 中 v6_full 的 +0.011 误判 |
