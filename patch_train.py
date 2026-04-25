import re

with open("/aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Modify get_pipeline_inputs: use instruction field instead of angle_bucket
old_prompt = '''    def get_pipeline_inputs(self, data):
        angle_value = "front view"
        if isinstance(data, dict):
            angle_value = data.get("angle_bucket", "front view")

        prompt = normalize_view_bucket(angle_value)'''

new_prompt = '''    def get_pipeline_inputs(self, data):
        # 优先使用 instruction 字段（如 "Rotate this object from front view to right side view."）
        # 保持训练和推理 prompt 格式一致
        prompt = None
        if isinstance(data, dict):
            prompt = data.get("instruction", None)
        if prompt is None:
            angle_value = "front view"
            if isinstance(data, dict):
                angle_value = data.get("angle_bucket", "front view")
            prompt = normalize_view_bucket(angle_value)'''

assert old_prompt in code, "Could not find old get_pipeline_inputs"
code = code.replace(old_prompt, new_prompt)

# 2. Add "instruction" to UnifiedDataset operators (next to angle_bucket)
old_operators = '            "angle_bucket": lambda x: x,'
new_operators = '            "angle_bucket": lambda x: x,\n            "instruction": lambda x: x,'

assert old_operators in code, "Could not find angle_bucket operator"
code = code.replace(old_operators, new_operators)

# 3. Add instruction fallback in JSONL remapping section
# Find the print block at end of remapping
old_print = '        print(f"[Info] Sample angle_bucket: {dataset.data[0].get(\'angle_bucket\')}")'
new_print = '''        print(f"[Info] Sample angle_bucket: {dataset.data[0].get('angle_bucket')}")
        print(f"[Info] Sample instruction: {dataset.data[0].get('instruction')}")'''

assert old_print in code, "Could not find sample angle_bucket print"
code = code.replace(old_print, new_print)

# 3b. Add instruction fallback before the print block
old_remap_marker = "    # ---- JSONL"
insert_instruction_block = '''    # ---- 确保 instruction 字段存在（训练 prompt 使用） ----
    _ANGLE_VIEW_MAP = {
        0: "front view", 45: "front-right view",
        90: "right side view", 135: "back-right view",
        180: "back view", 225: "back-left view",
        270: "left side view", 315: "front-left view",
    }

    '''
# We need to insert the instruction generation AFTER the main remapping loop
# Find the right spot: after the remapping for loop ends
old_after_remap = '''            remapped += 1
    if accelerator.is_main_process and remapped > 0:'''
new_after_remap = '''            remapped += 1
    # 确保每行都有 instruction 字段
    _ANGLE_VIEW_MAP = {
        0: "front view", 45: "front-right view",
        90: "right side view", 135: "back-right view",
        180: "back view", 225: "back-left view",
        270: "left side view", 315: "front-left view",
    }
    for row in dataset.data:
        if "instruction" not in row and "target_rotation_deg" in row:
            deg = int(row["target_rotation_deg"])
            view = _ANGLE_VIEW_MAP.get(deg, "front view")
            row["instruction"] = f"Rotate this object from front view to {view}."
    if accelerator.is_main_process and remapped > 0:'''

assert old_after_remap in code, "Could not find after remap section"
code = code.replace(old_after_remap, new_after_remap)

# 4. Update eval view_buckets to use instruction format
old_eval = '    eval_view_buckets = [normalize_view_bucket(x) for x in eval_view_buckets]'
new_eval = '''    # 使用 instruction 格式的 eval prompts（与训练一致）
    eval_view_buckets = [
        "Rotate this object from front view to front view.",
        "Rotate this object from front view to front-right view.",
        "Rotate this object from front view to right side view.",
        "Rotate this object from front view to back-right view.",
        "Rotate this object from front view to back view.",
        "Rotate this object from front view to back-left view.",
        "Rotate this object from front view to left side view.",
        "Rotate this object from front view to front-left view.",
    ]'''

assert old_eval in code, "Could not find eval_view_buckets normalization"
code = code.replace(old_eval, new_eval)

# 5. Remove the 8-unique check
old_check = '''    # Keep order but remove duplicates.
    eval_view_buckets = list(dict.fromkeys(eval_view_buckets))
    if len(eval_view_buckets) != 8:
        raise ValueError(
            "eval_view_buckets must contain exactly 8 unique view prompts. "
            f"Got {len(eval_view_buckets)}: {eval_view_buckets}"
        )'''
new_check = '    eval_view_buckets = list(dict.fromkeys(eval_view_buckets))'

assert old_check in code, "Could not find eval_view_buckets check"
code = code.replace(old_check, new_check)

# 6. Remove normalize_view_bucket in infer_rotation_sequence
old_infer_seq = '''        for view_prompt in view_buckets:
            view_prompt = normalize_view_bucket(view_prompt)'''
new_infer_seq = '''        for view_prompt in view_buckets:'''

assert old_infer_seq in code, "Could not find infer_rotation_sequence normalize"
code = code.replace(old_infer_seq, new_infer_seq)

with open("/aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.py", "w", encoding="utf-8") as f:
    f.write(code)

print("All patches applied successfully!")
