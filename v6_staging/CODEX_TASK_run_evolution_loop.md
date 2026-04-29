# Codex Task: Rewrite `run_evolution_loop.py` — v6 Baseline-First Multi-Dim Review

## Context

This is a VLM-feedback-driven 3D render quality evolution pipeline. The script runs on server `wwz` at `/aaaidata/zhangqisong/data_build/run_evolution_loop.py`.

**Core finding from v2–v5**: The v2 baseline (no actions taken) scored highest (0.7477 avg). The optimization loop hurt 8/10 objects. v6 redesigns the loop with:
- **Baseline stability confirmation** before any action
- **Three-zone decision** (reject → accepted → preserve → explore)
- **Bounded search** always from `baseline_state`, deterministic `preset_mode`
- **4-dim VLM diagnostics** (lighting + structure + color + physics)
- **9 exit reasons** (mutually exclusive, well-defined)

---

## New Interfaces You MUST Call (already written in other files)

### `review_object()` — new signature in `pipeline/stage5_5_vlm_review.py`

```python
def review_object(
    obj_id: str,
    renders_dir: str,
    output_dir: str,
    round_idx: int = 0,
    active_group: str = "lighting",
    prev_renders_dir: Optional[str] = None,   # pass None for baseline probes
    device: str = "cuda:0",
    history_file: Optional[str] = None,        # pass None for baseline probes
    prompt_appendix: str = "",
    issue_tags_whitelist: Optional[list] = None,
    reference_image_path: Optional[str] = None,  # NEW: path to T2I reference image
    profile_cfg: Optional[dict] = None,           # NEW: full profile dict
) -> dict:
```

**New fields in returned dict** (all default to `"good"` if VLM doesn't return them):
- `review["structure_consistency"]` → `"good" | "minor_mismatch" | "major_mismatch"`
- `review["color_consistency"]` → `"good" | "minor_shift" | "major_shift"`
- `review["physics_consistency"]` → `"good" | "minor_issue" | "major_issue"`

### `apply_feedback()` — new signature in `pipeline/stage5_6_feedback_apply.py`

```python
def apply_feedback(
    review_json_path: str,
    control_state_path: str,
    output_control_state_path: str,
    round_idx: int = 0,
    history_json_path: Optional[str] = None,
    hybrid_score: float = 0.0,
    action_blacklist: Optional[set] = None,
    preset_mode: bool = False,       # NEW: skip VLM suggestions, use deterministic selection
    active_group_override: Optional[str] = None,  # NEW: override active_group
) -> dict:
```

### `select_action()` — exposed in `pipeline/stage5_6_feedback_apply.py`

```python
def select_action(
    review: dict,
    active_group: str,
    control_state: dict,
    history: dict,
    aspace: dict,
    action_blacklist: Optional[set] = None,
    issue_tags: Optional[list] = None,
    preset_mode: bool = False,
) -> str:
```

---

## Task: Complete Rewrite of `run_evolution_loop.py`

### What to KEEP unchanged:
- All imports at top
- `_PIPELINE_STATE_PATH`, `_load_pipeline_state()`, `_save_pipeline_state()`, `_init_pipeline_state()`, `_update_obj_state()` — keep as-is
- `SCRIPT_DIR`, sys.path manipulation
- `BLENDER_BIN`, `RENDER_SCRIPT` constants
- `load_profile()` function
- `rerender_object()` function — keep exactly as-is
- `discover_obj_ids()` function
- `run_evolution()` function — keep mostly as-is (just pass `reference_image_path` and `profile_cfg` to `evolve_object`)
- `parse_args()` function
- `__main__` block

### What to CHANGE:

---

### Change 1a: `PROFILE_DEFAULTS` — add v6 fields

Replace the existing `PROFILE_DEFAULTS` dict with:

```python
PROFILE_DEFAULTS = {
    "dataset_name": "default",
    "task_type": "generic",
    "accept_threshold": 0.80,
    "reject_threshold": 0.40,
    "preserve_score_threshold": 0.77,
    "explore_threshold": 0.68,
    "stability_threshold": 0.03,
    "unstable_variance_limit": 0.05,
    "mid_budget": 1,
    "low_budget": 2,
    "improve_eps": 0.01,
    "max_rounds": 5,
    "patience": 4,
    "review_view_policy": "canonical_4",
    "action_whitelist": None,
    "issue_tags_whitelist": None,
    "prompt_appendix": "",
    "reference_images_dir": None,
}
```

Also **remove** the old standalone constants at module level:
```python
# DELETE THESE:
MAX_ROUNDS = 3
ACCEPT_TH = 0.80
IMPROVE_EPS = 0.02
PATIENCE = 2
REJECT_TH = 0.40
```
(they are now in PROFILE_DEFAULTS)

Keep these module-level constants (they are not in profile):
```python
MESH_EVIDENCE_TAGS = {...}
DIAGNOSIS_TO_TAG = {...}
GROUP_ORDER = [...]
BLENDER_BIN = ...
RENDER_SCRIPT = ...
```

---

### Change 1b: New helper functions (add after `load_profile`)

Add these three new functions:

```python
def _resolve_reference_image(obj_id: str, profile_cfg: dict, renders_dir: str) -> Optional[str]:
    """Resolve T2I reference image path. Returns absolute path or None."""
    ref_dir = profile_cfg.get("reference_images_dir", None)
    if ref_dir is None:
        # Default: renders_dir/../images
        ref_dir = os.path.join(os.path.dirname(os.path.abspath(renders_dir)), "images")
    if not os.path.isabs(ref_dir):
        ref_dir = os.path.join(SCRIPT_DIR, ref_dir)
    ref_path = os.path.join(ref_dir, f"{obj_id}.png")
    return ref_path if os.path.exists(ref_path) else None


def _aggregate_baseline_diagnostics(diag_list: list) -> dict:
    """
    Aggregate diagnostics across 2–3 baseline probes.
    Rules:
    - structure_consistency: worst-case (major_mismatch > minor_mismatch > good)
    - physics_consistency:   worst-case (major_issue > minor_issue > good)
    - color_consistency:     majority vote
    - lighting_diagnosis:    majority vote
    """
    if not diag_list:
        return {}

    # Worst-case: structure
    struct_vals = [d.get("structure_consistency", "good") for d in diag_list]
    if "major_mismatch" in struct_vals:
        agg_structure = "major_mismatch"
    elif "minor_mismatch" in struct_vals:
        agg_structure = "minor_mismatch"
    else:
        agg_structure = "good"

    # Worst-case: physics
    phys_vals = [d.get("physics_consistency", "good") for d in diag_list]
    if "major_issue" in phys_vals:
        agg_physics = "major_issue"
    elif "minor_issue" in phys_vals:
        agg_physics = "minor_issue"
    else:
        agg_physics = "good"

    # Majority vote: color
    color_vals = [d.get("color_consistency", "good") for d in diag_list]
    agg_color = Counter(color_vals).most_common(1)[0][0]

    # Majority vote: lighting_diagnosis
    light_vals = [d.get("lighting_diagnosis", "good") for d in diag_list]
    agg_lighting = Counter(light_vals).most_common(1)[0][0]

    return {
        "structure_consistency": agg_structure,
        "physics_consistency": agg_physics,
        "color_consistency": agg_color,
        "lighting_diagnosis": agg_lighting,
    }


def _determine_active_group(review: dict) -> str:
    """
    Deterministically pick the active group from review diagnostics.
    Priority: lighting > material > object > scene
    """
    if review.get("lighting_diagnosis", "good") != "good":
        return "lighting"
    if review.get("color_consistency", "good") != "good":
        return "material"
    if review.get("physics_consistency", "good") == "minor_issue":
        return "object"
    # Default fallback
    return "lighting"
```

---

### Change 1c: Updated `blacklist_failed_action`

Replace the existing `blacklist_failed_action` with this updated version:

```python
def blacklist_failed_action(action_blacklist: set, aspace: dict, action_name: str,
                             sub_results=None) -> list:
    """
    Blacklist a failed action and its applied sub-actions.
    sub_results: list of ActionResult objects from apply_feedback (can be None)
    Returns list of newly blocked action names.
    """
    if not action_name or action_name == "NO_OP":
        return []
    blocked = {action_name}
    compound = aspace.get("compound_actions", {}).get(action_name)
    if compound:
        # Always blacklist compound itself
        # Also blacklist sub-actions that were actually applied
        if sub_results:
            for sr in sub_results:
                if getattr(sr, "applied", False):
                    blocked.add(getattr(sr, "action", getattr(sr, "action_name", "")))
        else:
            # No sub_results info: blacklist all sub-actions conservatively
            blocked.update(compound.get("sub_actions", []) or [])
    newly_blocked = sorted(a for a in blocked if a not in action_blacklist)
    action_blacklist.update(blocked)
    return newly_blocked
```

---

### Change 1d: Complete rewrite of `evolve_object()`

This is the main change. Replace the entire `evolve_object()` function body with the v6 logic below.

**New function signature**:
```python
def evolve_object(obj_id, renders_dir, evolution_dir, meshes_dir=None,
                  device="cuda:0", max_rounds=None, blender_bin=BLENDER_BIN,
                  profile=None, reference_image_path=None):
```

**Complete new function body**:

```python
def evolve_object(obj_id, renders_dir, evolution_dir, meshes_dir=None,
                  device="cuda:0", max_rounds=None, blender_bin=BLENDER_BIN,
                  profile=None, reference_image_path=None):
    prof = profile or PROFILE_DEFAULTS

    # --- Extract thresholds from profile ---
    _ACCEPT_TH     = float(prof.get("accept_threshold",      PROFILE_DEFAULTS["accept_threshold"]))
    _REJECT_TH     = float(prof.get("reject_threshold",      PROFILE_DEFAULTS["reject_threshold"]))
    _PRESERVE_TH   = float(prof.get("preserve_score_threshold", PROFILE_DEFAULTS["preserve_score_threshold"]))
    _EXPLORE_TH    = float(prof.get("explore_threshold",     PROFILE_DEFAULTS["explore_threshold"]))
    _STABILITY_TH  = float(prof.get("stability_threshold",   PROFILE_DEFAULTS["stability_threshold"]))
    _UNSTABLE_VAR  = float(prof.get("unstable_variance_limit", PROFILE_DEFAULTS["unstable_variance_limit"]))
    _MID_BUDGET    = int(  prof.get("mid_budget",             PROFILE_DEFAULTS["mid_budget"]))
    _LOW_BUDGET    = int(  prof.get("low_budget",             PROFILE_DEFAULTS["low_budget"]))
    _IMPROVE_EPS   = float(prof.get("improve_eps",            PROFILE_DEFAULTS["improve_eps"]))
    _MAX_ROUNDS    = int(  prof.get("max_rounds",             max_rounds or PROFILE_DEFAULTS["max_rounds"]))
    _PATIENCE      = int(  prof.get("patience",               PROFILE_DEFAULTS["patience"]))
    _PROMPT_APPENDIX = prof.get("prompt_appendix", "")
    _ISSUE_TAGS_WL   = prof.get("issue_tags_whitelist", None)

    # --- Setup directories ---
    obj_dir    = os.path.join(evolution_dir, obj_id)
    reviews_dir = os.path.join(obj_dir, "reviews")
    states_dir  = os.path.join(obj_dir, "states")
    history_json = os.path.join(states_dir, "history.json")
    sharp_hist   = os.path.join(obj_dir, "sharpness_history.json")
    os.makedirs(reviews_dir, exist_ok=True)
    os.makedirs(states_dir, exist_ok=True)

    aspace = load_action_space()
    control_state = default_control_state(aspace)
    action_blacklist: set = set()
    state_log: list = []

    # =========================================================
    # PHASE 1: Baseline Stability Confirmation
    # Run 2–3 clean probes (no prev_renders_dir, no history write)
    # to confirm the baseline score is stable.
    # =========================================================

    baseline_probes: list = []   # list of (score, review_dict)
    STABILITY_TH = _STABILITY_TH
    UNSTABLE_VAR = _UNSTABLE_VAR

    def _run_clean_probe(probe_idx_offset: int) -> Optional[tuple]:
        """Run a single clean baseline probe. Returns (hybrid_score, review_dict) or None."""
        cs_path = os.path.join(states_dir, f"cs_baseline_p{probe_idx_offset:02d}.json")
        save_control_state(control_state, cs_path)
        rev = review_object(
            obj_id=obj_id,
            renders_dir=renders_dir,
            output_dir=reviews_dir,
            round_idx=probe_idx_offset,
            active_group="lighting",
            prev_renders_dir=None,     # clean: no pairwise context
            device=device,
            history_file=None,         # clean: no sharpness history write
            prompt_appendix=_PROMPT_APPENDIX,
            issue_tags_whitelist=_ISSUE_TAGS_WL,
            reference_image_path=reference_image_path,
            profile_cfg=prof,
        )
        if not rev:
            return None
        return (rev.get("hybrid_score", 0.0), rev)

    # Probe 0 — always
    p0 = _run_clean_probe(0)
    if p0 is None:
        print(f"  [{obj_id}] Baseline probe 0 failed — render_hard_fail")
        return {
            "obj_id": obj_id, "final_hybrid": 0.0, "probes_run": 0,
            "updates_run": 0, "accepted": False,
            "exit_reason": "rejected_render_hard_fail",
            "state_log": [], "best_state": control_state, "best_state_path": "",
        }
    baseline_probes.append(p0)
    print(f"  [{obj_id}] Baseline p0: {p0[0]:.4f}")

    # Probe 1 — always
    p1 = _run_clean_probe(1)
    if p1 is None:
        print(f"  [{obj_id}] Baseline probe 1 failed — render_hard_fail")
        return {
            "obj_id": obj_id, "final_hybrid": p0[0], "probes_run": 1,
            "updates_run": 0, "accepted": False,
            "exit_reason": "rejected_render_hard_fail",
            "state_log": [], "best_state": control_state, "best_state_path": "",
        }
    baseline_probes.append(p1)
    print(f"  [{obj_id}] Baseline p1: {p1[0]:.4f}")

    scores_so_far = [p0[0], p1[0]]
    score_repeats = 2

    # Probe 2 — only if |p0 - p1| > STABILITY_TH
    if abs(p0[0] - p1[0]) > STABILITY_TH:
        p2 = _run_clean_probe(2)
        if p2 is not None:
            baseline_probes.append(p2)
            scores_so_far.append(p2[0])
            score_repeats = 3
            print(f"  [{obj_id}] Baseline p2 (stability check): {p2[0]:.4f}")

    # Compute confirmed baseline score
    import statistics as _stats
    if score_repeats == 3:
        confirmed_score = _stats.median(scores_so_far)
        variance = _stats.variance(scores_so_far)
        if variance > UNSTABLE_VAR:
            print(f"  [{obj_id}] UNSTABLE: variance={variance:.4f} > {UNSTABLE_VAR} — rejected_unstable_score")
            return {
                "obj_id": obj_id, "final_hybrid": confirmed_score, "probes_run": score_repeats,
                "updates_run": 0, "accepted": False,
                "exit_reason": "rejected_unstable_score",
                "state_log": [], "best_state": control_state, "best_state_path": "",
            }
    else:
        confirmed_score = sum(scores_so_far) / len(scores_so_far)   # mean of p0, p1

    print(f"  [{obj_id}] Confirmed baseline: {confirmed_score:.4f} (from {score_repeats} probes)")

    # Aggregate diagnostics across baseline probes (worst-case / majority)
    baseline_review_dicts = [bp[1] for bp in baseline_probes]
    agg_baseline_diag = _aggregate_baseline_diagnostics(baseline_review_dicts)
    confirmed_structure = agg_baseline_diag.get("structure_consistency", "good")
    confirmed_physics   = agg_baseline_diag.get("physics_consistency",   "good")
    confirmed_color     = agg_baseline_diag.get("color_consistency",     "good")
    confirmed_lighting  = agg_baseline_diag.get("lighting_diagnosis",    "good")
    print(f"  [{obj_id}] Diagnostics: struct={confirmed_structure} phys={confirmed_physics} "
          f"color={confirmed_color} light={confirmed_lighting}")

    best_score = confirmed_score
    best_state = copy.deepcopy(control_state)
    baseline_state = copy.deepcopy(control_state)   # v6: always reset from here

    # =========================================================
    # PHASE 2: Three-Zone Decision
    # Priority order: reject → accepted_baseline → preserved_high_zone
    #                → rejected_low → mid/low search
    # =========================================================

    def _make_result(exit_reason: str, final_score: float, probes: int, updates: int) -> dict:
        best_state_path = os.path.join(states_dir, "control_state_best.json")
        save_control_state(best_state, best_state_path)
        result = {
            "obj_id": obj_id,
            "final_hybrid": final_score,
            "probes_run": probes + score_repeats,
            "updates_run": updates,
            "accepted": (final_score >= _ACCEPT_TH),
            "exit_reason": exit_reason,
            "state_log": state_log,
            "best_state": best_state,
            "best_state_path": best_state_path,
            "confirmed_score": confirmed_score,
            "diagnostics": agg_baseline_diag,
        }
        result_path = os.path.join(obj_dir, "evolution_result.json")
        with open(result_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "best_state"}, f, indent=2)
        _fs = final_score
        print(f"\n[{obj_id}] Exit: {exit_reason} | final={_fs:.4f} | "
              f"accepted={result['accepted']} | updates={updates}")
        return result

    # Step 1: Reject — structure/physics serious issues
    if confirmed_structure == "major_mismatch":
        return _make_result("rejected_mesh", confirmed_score, 0, 0)
    if confirmed_physics == "major_issue":
        return _make_result("rejected_physics_major", confirmed_score, 0, 0)

    # Step 2: accepted_baseline — score already high enough
    if confirmed_score >= _ACCEPT_TH:
        return _make_result("accepted_baseline", confirmed_score, 0, 0)

    # Step 3: preserved_high_zone — score in [preserve_th, accept_th)
    if confirmed_score >= _PRESERVE_TH:
        return _make_result("preserved_high_zone", confirmed_score, 0, 0)

    # Step 4: rejected_low_quality — score below minimum
    if confirmed_score < _REJECT_TH:
        return _make_result("rejected_low_quality", confirmed_score, 0, 0)

    # Step 5: Determine zone and budget
    if confirmed_score >= _EXPLORE_TH:
        zone = "mid"
        budget = _MID_BUDGET
    else:
        zone = "low"
        budget = _LOW_BUDGET

    print(f"  [{obj_id}] Zone={zone}, budget={budget}, exploring with preset_mode...")

    # =========================================================
    # PHASE 3: Bounded Search
    # Each attempt resets to baseline_state.
    # Active group determined deterministically from diagnostics.
    # =========================================================

    # Determine active group from aggregated diagnostics
    active_group = _determine_active_group(agg_baseline_diag)

    # Build a combined review dict from aggregated diagnostics for apply_feedback
    # We use the last baseline probe's full review and overlay aggregated diagnostics
    combined_review = dict(baseline_review_dicts[-1])
    combined_review.update(agg_baseline_diag)

    # Save aggregated review for apply_feedback to consume
    agg_review_path = os.path.join(reviews_dir, f"{obj_id}_baseline_agg.json")
    with open(agg_review_path, "w") as f:
        json.dump(combined_review, f, indent=2)

    for attempt_idx in range(budget):
        print(f"\n  [{obj_id}] Search attempt {attempt_idx+1}/{budget} | "
              f"group={active_group} | zone={zone}")

        # Reset to baseline state for each attempt
        attempt_state = copy.deepcopy(baseline_state)
        cs_attempt_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}.json")
        save_control_state(attempt_state, cs_attempt_path)

        next_cs_path = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}_after.json")

        feedback = apply_feedback(
            review_json_path=agg_review_path,
            control_state_path=cs_attempt_path,
            output_control_state_path=next_cs_path,
            round_idx=attempt_idx,
            history_json_path=history_json,
            hybrid_score=confirmed_score,
            action_blacklist=action_blacklist,
            preset_mode=True,
            active_group_override=active_group,
        )

        action = feedback.get("action_taken", "NO_OP")
        sub_results = feedback.get("sub_results", None)

        # Check if compound yielded all-skipped (treat as NO_OP)
        if action != "NO_OP" and sub_results is not None:
            if not any(getattr(sr, "applied", False) for sr in sub_results):
                print(f"  [{obj_id}] Action '{action}' all sub-actions skipped → NO_OP")
                action = "NO_OP"

        if action in ("NO_OP", "__MESH_REJECT__"):
            print(f"  [{obj_id}] No actionable preset for attempt {attempt_idx+1}: {action}")
            blacklist_failed_action(action_blacklist, aspace, action, sub_results)
            log_entry = {
                "attempt": attempt_idx, "zone": zone, "action_taken": action,
                "hybrid_score": confirmed_score, "exit_reason": "no_action",
            }
            state_log.append(log_entry)
            continue

        attempt_control_state = feedback["control_state"]

        # Re-render with the new control state
        if meshes_dir:
            rerender_out_base = os.path.join(obj_dir, f"renders_a{attempt_idx:02d}")
            success = rerender_object(
                obj_id=obj_id,
                meshes_dir=meshes_dir,
                output_dir=rerender_out_base,
                control_state=attempt_control_state,
                blender_bin=blender_bin,
            )
            attempt_renders_dir_candidate = rerender_out_base
            renders_ok = (
                success
                and os.path.isdir(rerender_out_base)
                and any(f.endswith(".png") for f in os.listdir(rerender_out_base))
            )
            if not renders_ok:
                print(f"  [{obj_id}] Re-render failed for attempt {attempt_idx+1}, skip.")
                blacklist_failed_action(action_blacklist, aspace, action, sub_results)
                state_log.append({
                    "attempt": attempt_idx, "zone": zone, "action_taken": action,
                    "hybrid_score": confirmed_score, "exit_reason": "render_failed",
                })
                continue
            attempt_renders_dir = attempt_renders_dir_candidate
        else:
            print(f"  [{obj_id}] No meshes-dir, dry run attempt {attempt_idx+1}")
            attempt_renders_dir = renders_dir

        # Evaluate the attempted render (use normal review with prev context)
        attempt_cs_save = os.path.join(states_dir, f"cs_attempt_{attempt_idx:02d}_eval.json")
        save_control_state(attempt_control_state, attempt_cs_save)

        attempt_review = review_object(
            obj_id=obj_id,
            renders_dir=attempt_renders_dir,
            output_dir=reviews_dir,
            round_idx=score_repeats + attempt_idx,
            active_group=active_group,
            prev_renders_dir=renders_dir,   # compare vs original baseline renders
            device=device,
            history_file=sharp_hist,
            prompt_appendix=_PROMPT_APPENDIX,
            issue_tags_whitelist=_ISSUE_TAGS_WL,
            reference_image_path=reference_image_path,
            profile_cfg=prof,
        )

        if not attempt_review:
            print(f"  [{obj_id}] Attempt review failed, skip.")
            blacklist_failed_action(action_blacklist, aspace, action, sub_results)
            state_log.append({
                "attempt": attempt_idx, "zone": zone, "action_taken": action,
                "hybrid_score": confirmed_score, "exit_reason": "review_failed",
            })
            continue

        attempt_score = attempt_review.get("hybrid_score", 0.0)
        print(f"  [{obj_id}] Attempt {attempt_idx+1}: action={action} → score={attempt_score:.4f} "
              f"(baseline={confirmed_score:.4f}, delta={attempt_score-confirmed_score:+.4f})")

        log_entry = {
            "attempt": attempt_idx, "zone": zone, "action_taken": action,
            "hybrid_score": attempt_score,
            "delta_vs_confirmed": attempt_score - confirmed_score,
            "exit_reason": None,
            "sub_actions_applied": (
                [getattr(sr, "action", "") for sr in sub_results if getattr(sr, "applied", False)]
                if sub_results else None
            ),
        }

        if attempt_score > confirmed_score + _IMPROVE_EPS:
            # Success: accepted_after_try
            best_score = attempt_score
            best_state = copy.deepcopy(attempt_control_state)
            log_entry["exit_reason"] = "accepted_after_try"
            state_log.append(log_entry)
            return _make_result("accepted_after_try", best_score, attempt_idx + 1, attempt_idx + 1)
        else:
            # Did not improve: blacklist and continue (if budget allows)
            blacklist_failed_action(action_blacklist, aspace, action, sub_results)
            log_entry["exit_reason"] = "attempt_failed"
            state_log.append(log_entry)
            print(f"  [{obj_id}] Attempt {attempt_idx+1} did not improve sufficiently, blacklisting '{action}'")

    # Budget exhausted
    if zone == "mid":
        exit_reason = "mid_no_improve"
    else:
        exit_reason = "low_exhausted"

    return _make_result(exit_reason, confirmed_score, budget, 0)
```

---

### Change 1e: Update `run_evolution()` to resolve reference image and pass to `evolve_object`

In `run_evolution()`, add reference image resolution before the loop, and pass it to `evolve_object`:

```python
def run_evolution(renders_dir, output_dir, obj_ids=None, meshes_dir=None,
                  device="cuda:0", max_rounds=None, blender_bin=BLENDER_BIN, profile=None):
    os.makedirs(output_dir, exist_ok=True)
    if obj_ids is None:
        obj_ids = discover_obj_ids(renders_dir)
    prof = profile or PROFILE_DEFAULTS
    print(f"[Evolution] Processing {len(obj_ids)} objects: {obj_ids}")
    all_results = {}
    _pstate = _init_pipeline_state(obj_ids, output_dir)
    _save_pipeline_state(_pstate)

    for obj_id in obj_ids:
        try:
            # Resolve reference image for this object
            ref_img = _resolve_reference_image(obj_id, prof, renders_dir)
            if ref_img:
                print(f"[Evolution] Reference image for {obj_id}: {ref_img}")
            else:
                print(f"[Evolution] No reference image found for {obj_id}")

            result = evolve_object(
                obj_id=obj_id,
                renders_dir=renders_dir,
                evolution_dir=output_dir,
                meshes_dir=meshes_dir,
                device=device,
                max_rounds=max_rounds,
                blender_bin=blender_bin,
                profile=profile,
                reference_image_path=ref_img,   # NEW
            )
            all_results[obj_id] = result
            _update_obj_state(_pstate, obj_id, result)
        except Exception as e:
            print(f"[Evolution] ERROR on {obj_id}: {e}")
            import traceback
            traceback.print_exc()
            all_results[obj_id] = {"obj_id": obj_id, "error": str(e)}
            _update_obj_state(_pstate, obj_id, {"final_hybrid": 0, "exit_reason": "error",
                                                  "accepted": False, "probes_run": 0})

    accepted = sum(1 for r in all_results.values() if r.get("accepted", False))
    final_scores = {k: v.get("final_hybrid", 0.0) for k, v in all_results.items()}
    summary = {
        "total": len(all_results),
        "accepted": accepted,
        "acceptance_rate": accepted / max(1, len(all_results)),
        "final_scores": final_scores,
        "results": all_results,
    }
    device_tag = device.replace(":", "").replace(",", "_")
    partial_path = os.path.join(output_dir, f"_partial_{device_tag}.json")
    partial = {
        "device": device,
        "obj_ids": list(all_results.keys()),
        "accepted": accepted,
        "total": len(all_results),
        "final_scores": final_scores,
    }
    with open(partial_path, "w") as f:
        json.dump(partial, f, indent=2)

    print(f"\n[Evolution] Done: {accepted}/{len(all_results)} accepted, "
          f"avg_score={sum(final_scores.values())/max(1,len(final_scores)):.3f}")
    return summary
```

---

## Full Existing File (for context)

Below is the current v5 `run_evolution_loop.py` that you need to modify:

```python
"""
run_evolution_loop.py — VLM-feedback-driven render self-evolution.
"""

import os
import sys
import json
import copy
import shutil
import argparse
import subprocess
from collections import Counter
from pathlib import Path
from typing import Optional, List

# ─── v5: PIPELINE_STATE.json persistence ────────────────────────────────────
import datetime as _dt

_PIPELINE_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PIPELINE_STATE.json")

def _load_pipeline_state():
    if os.path.exists(_PIPELINE_STATE_PATH):
        with open(_PIPELINE_STATE_PATH) as f:
            return json.load(f)
    return None

def _save_pipeline_state(state):
    state["timestamp"] = _dt.datetime.now().isoformat()
    with open(_PIPELINE_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def _init_pipeline_state(obj_ids, output_dir):
    return {
        "task": f"evolution_{os.path.basename(output_dir)}",
        "phase": "running",
        "status": "in_progress",
        "codex_review_threadId": None,
        "objects": {oid: {"status": "pending"} for oid in obj_ids},
        "timestamp": _dt.datetime.now().isoformat(),
    }

def _update_obj_state(pstate, obj_id, result):
    pstate["objects"][obj_id] = {
        "status": "done",
        "final_score": result.get("final_hybrid", 0),
        "exit_reason": result.get("exit_reason", "unknown"),
        "accepted": result.get("accepted", False),
        "probes_run": result.get("probes_run", 0),
    }
    done = sum(1 for v in pstate["objects"].values() if v.get("status") == "done")
    total = len(pstate["objects"])
    if done >= total:
        pstate["status"] = "completed"
    _save_pipeline_state(pstate)

# ─── end v5 persistence ────────────────────────────────────────────────────


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "pipeline"))

from stage5_5_vlm_review import review_object
from stage5_6_feedback_apply import (
    apply_feedback, load_action_space, default_control_state,
    save_control_state, ActionResult,
)

MAX_ROUNDS = 3
ACCEPT_TH = 0.80
IMPROVE_EPS = 0.02
PATIENCE = 2
REJECT_TH = 0.40

# Mesh-level reject evidence tags
MESH_EVIDENCE_TAGS = {
    "mesh_interpenetration",
    "geometry_distortion",
    "ground_intersection",
}

# diagnosis → issue_tag mapping
DIAGNOSIS_TO_TAG = {
    "flat_no_rim": "flat_lighting",
    "flat_low_contrast": "weak_subject_separation",
    "underexposed_global": "underexposed",
    "underexposed_shadow": "harsh_shadow",
    "harsh_shadow_key": "harsh_shadow",
}

GROUP_ORDER = ["lighting", "camera", "object", "scene", "material"]

BLENDER_BIN = os.environ.get("BLENDER_BIN", "/home/wuwenzhuo/blender-4.24/blender")
RENDER_SCRIPT = os.path.join(SCRIPT_DIR, "pipeline", "stage4_blender_render.py")

PROFILE_DEFAULTS = {
    "dataset_name": "default",
    "task_type": "generic",
    "accept_threshold": ACCEPT_TH,
    "reject_threshold": REJECT_TH,
    "max_rounds": MAX_ROUNDS,
    "patience": PATIENCE,
    "improve_eps": IMPROVE_EPS,
    "explore_threshold": 0.68,
    "review_view_policy": "canonical_4",
    "action_whitelist": None,
    "issue_tags_whitelist": None,
    "prompt_appendix": "",
}


def load_profile(path):
    if path is None:
        return dict(PROFILE_DEFAULTS)
    with open(path) as f:
        data = json.load(f)
    return {**PROFILE_DEFAULTS, **data}


def classify_score_zone(score, preserve_threshold, explore_threshold):
    if score >= preserve_threshold:
        return "high"
    if score >= explore_threshold:
        return "mid"
    return "low"


def blacklist_failed_action(action_blacklist, aspace, action_name, sub_actions=None):
    if not action_name or action_name == "NO_OP":
        return []
    blocked = {action_name}
    compound = aspace.get("compound_actions", {}).get(action_name)
    if compound:
        blocked.update(compound.get("sub_actions", []) or [])
    if sub_actions:
        blocked.update(sub_actions)
    newly_blocked = sorted(a for a in blocked if a not in action_blacklist)
    action_blacklist.update(blocked)
    return newly_blocked


def pick_next_group(issue_tags, suggested_actions, group_tried, aspace):
    fallback_map = aspace.get("issue_to_action_fallback", {})
    compound_actions = aspace.get("compound_actions", {})
    candidate_groups = []
    for tag in issue_tags:
        for action in fallback_map.get(tag, []):
            if action in compound_actions:
                grp = compound_actions[action].get("group")
                if grp and grp not in group_tried:
                    candidate_groups.append(grp)
            for grp, gdata in aspace["groups"].items():
                if action in gdata["actions"] and grp not in group_tried:
                    candidate_groups.append(grp)
    for action in suggested_actions:
        if action in compound_actions:
            grp = compound_actions[action].get("group")
            if grp and grp not in group_tried:
                candidate_groups.append(grp)
        for grp, gdata in aspace["groups"].items():
            if action in gdata["actions"] and grp not in group_tried:
                candidate_groups.append(grp)
    if candidate_groups:
        return Counter(candidate_groups).most_common(1)[0][0]
    for grp in GROUP_ORDER:
        if grp not in group_tried:
            return grp
    return GROUP_ORDER[0]


def rerender_object(obj_id, meshes_dir, output_dir, control_state,
                    blender_bin=BLENDER_BIN, resolution=512, engine="EEVEE"):
    glb_path = os.path.join(meshes_dir, f"{obj_id}.glb")
    if not os.path.exists(glb_path):
        print(f"  [rerender] GLB not found: {glb_path}")
        return False
    cs_tmp = os.path.join(output_dir, f"_cs_{obj_id}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(cs_tmp, "w") as f:
        json.dump(control_state, f, indent=2)
    cmd = [
        blender_bin, "-b", "-P", RENDER_SCRIPT, "--",
        "--input-dir", meshes_dir,
        "--output-dir", output_dir,
        "--resolution", str(resolution),
        "--engine", engine,
        "--obj-id", obj_id,
        "--control-state", cs_tmp,
    ]
    print(f"  [rerender] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [rerender] Blender failed:\n{result.stderr[-1000:]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  [rerender] Blender timed out")
        return False
    except Exception as e:
        print(f"  [rerender] Error: {e}")
        return False


def evolve_object(obj_id, renders_dir, evolution_dir, meshes_dir=None,
                  device="cuda:0", max_rounds=MAX_ROUNDS, blender_bin=BLENDER_BIN,
                  profile=None):
    # ... [full existing 300-line body — SEE ABOVE IN CONTEXT SUMMARY]
    pass


def discover_obj_ids(renders_dir):
    return sorted(
        d for d in os.listdir(renders_dir)
        if os.path.isdir(os.path.join(renders_dir, d))
    )


def run_evolution(renders_dir, output_dir, obj_ids=None, meshes_dir=None,
                  device="cuda:0", max_rounds=MAX_ROUNDS, blender_bin=BLENDER_BIN, profile=None):
    # ... [existing body — REPLACE with updated version above]
    pass


def parse_args():
    p = argparse.ArgumentParser(description="VLM-feedback render self-evolution loop")
    p.add_argument("--renders-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--obj-ids", nargs="+", default=None)
    p.add_argument("--meshes-dir", default=None)
    p.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--blender", default=BLENDER_BIN, dest="blender_bin")
    p.add_argument("--profile", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = load_profile(args.profile)
    if args.profile:
        print(f"[Evolution] Profile: {args.profile}")
    summary = run_evolution(
        renders_dir=args.renders_dir,
        output_dir=args.output_dir,
        obj_ids=args.obj_ids,
        meshes_dir=args.meshes_dir,
        device=args.device,
        max_rounds=args.max_rounds,
        blender_bin=args.blender_bin,
        profile=profile,
    )
    print(f"\n=== Summary: {summary['accepted']}/{summary['total']} accepted, "
          f"rate={summary['acceptance_rate']:.1%} ===")
```

---

## Summary Checklist

- [ ] `PROFILE_DEFAULTS` updated with all v6 fields; old module-level constants removed
- [ ] `_resolve_reference_image()` added
- [ ] `_aggregate_baseline_diagnostics()` added
- [ ] `_determine_active_group()` added
- [ ] `blacklist_failed_action()` updated (sub_results instead of sub_actions list)
- [ ] `evolve_object()` completely rewritten (Phases 1–3)
- [ ] `run_evolution()` updated to call `_resolve_reference_image` and pass `reference_image_path`
- [ ] `import statistics as _stats` included inside `evolve_object` (or at top of file)
- [ ] `classify_score_zone()` and `pick_next_group()` can remain (not called in new code but harmless to keep)

---

## Verification Commands

```bash
cd /aaaidata/zhangqisong/data_build

# 1. Syntax check
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -m py_compile run_evolution_loop.py
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -m py_compile pipeline/stage5_5_vlm_review.py
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -m py_compile pipeline/stage5_6_feedback_apply.py

# 2. Unit tests for select_action
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 -c "
import sys; sys.path.insert(0, 'pipeline')
from stage5_6_feedback_apply import select_action, load_action_space
aspace = load_action_space()

# Test 1: preset_mode=True, diagnosis=flat_no_rim → L_RIM_UP
review = {'lighting_diagnosis': 'flat_no_rim', 'issue_tags': ['flat_lighting']}
cs = {'lighting': {'rim_scale': 1.0, 'fill_scale': 1.0}}
action = select_action(review, 'lighting', cs, {}, aspace, preset_mode=True)
assert action == 'L_RIM_UP', f'T1 FAIL: got {action}'

# Test 2: preset_mode=True, color_shift → M_SATURATION_DOWN
review2 = {'lighting_diagnosis': 'good', 'color_consistency': 'minor_shift', 'issue_tags': []}
action2 = select_action(review2, 'material', cs, {}, aspace, preset_mode=True)
assert action2 == 'M_SATURATION_DOWN', f'T2 FAIL: got {action2}'

# Test 3: preset_mode=True, physics minor → O_LOWER
review3 = {'lighting_diagnosis': 'good', 'physics_consistency': 'minor_issue', 'issue_tags': ['floating_object']}
action3 = select_action(review3, 'object', cs, {}, aspace, preset_mode=True)
assert action3 == 'O_LOWER', f'T3 FAIL: got {action3}'

# Test 4: preset_mode=False → walks VLM suggestion path
review4 = {'lighting_diagnosis': 'flat_no_rim', 'issue_tags': ['flat_lighting'],
           'suggested_actions': ['L_FILL_DOWN']}
action4 = select_action(review4, 'lighting', cs, {}, aspace, preset_mode=False)
assert action4 == 'L_FILL_DOWN', f'T4 FAIL: got {action4}'

print('ALL TESTS PASSED')
"

# 3. Smoke test (3 objects, dry run without meshes-dir)
/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3 run_evolution_loop.py \
  --renders-dir pipeline/data/renders \
  --output-dir pipeline/data/evolution_v6_smoke \
  --obj-ids obj_007 obj_009 obj_010 \
  --device cuda:0 \
  --profile configs/dataset_profiles/rotation_v6.json
```

**Smoke test acceptance criteria**:
- obj_009: exit `preserved_high_zone` or `accepted_baseline` (was 0.78 in v2)
- obj_010: exit `accepted_baseline` or `accepted_after_try` (was 0.83 in v2)
- obj_007: exit `mid_no_improve` or `accepted_after_try` (was 0.75 in v2)
- All objects: `final_score >= confirmed_score - 0.01`
- Review JSON contains `structure_consistency`, `color_consistency`, `physics_consistency`
