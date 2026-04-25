"""
Build new object assets from stage1 through stage3 into an isolated sandbox root.

Outputs:
- prompts.json
- images/
- images_rgba/
- meshes_raw/
- meshes/
- summary.json

The existing global pipeline/data roots remain untouched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
STAGE1_SCRIPT = REPO_ROOT / "pipeline" / "stage1_text_expansion.py"
STAGE2_SCRIPT = REPO_ROOT / "pipeline" / "stage2_t2i_generate.py"
STAGE25_SCRIPT = REPO_ROOT / "pipeline" / "stage2_5_sam2_segment.py"
STAGE3_SCRIPT = REPO_ROOT / "pipeline" / "stage3_image_to_3d.py"

DEFAULT_OBJECTS_FILE = REPO_ROOT / "configs" / "seed_concepts" / "scene_obj021_050_objects.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "pipeline" / "data" / "dataset_scene_v7_obj021_obj050_stage1_assets_20260408"


def parse_args():
    parser = argparse.ArgumentParser(description="Build obj021-050 assets from stage1 into an isolated sandbox")
    parser.add_argument("--objects-file", default=str(DEFAULT_OBJECTS_FILE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--stage1-mode", choices=["api", "template-only", "dry-run"], default="template-only")
    parser.add_argument("--stage1-model", default="claude-haiku-4-5")
    parser.add_argument("--t2i-device", default="cuda:0")
    parser.add_argument("--sam-device", default="cuda:0")
    parser.add_argument("--i23d-devices", default="cuda:0,cuda:1,cuda:2")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--shape-only", action="store_true")
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--launch-stagger-seconds", type=float, default=1.0)
    return parser.parse_args()


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_step(cmd: List[str], name: str) -> dict:
    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "name": name,
        "command": cmd,
        "return_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": completed.stdout[-8000:],
    }


def launch_step(cmd: List[str], name: str) -> dict:
    started = time.time()
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "name": name,
        "command": cmd,
        "process": process,
        "started_at": started,
    }


def run_sharded_workers(
    *,
    name_prefix: str,
    script_path: Path,
    python_bin: str,
    devices: List[str],
    obj_ids: List[str],
    base_args: List[str],
    no_skip: bool,
    launch_stagger_seconds: float,
) -> List[dict]:
    launched = []
    for index, shard in enumerate(shard_ids(obj_ids, devices)):
        cmd = [
            python_bin,
            str(script_path),
            "--device",
            shard["device"],
            *base_args,
            "--ids",
            ",".join(shard["obj_ids"]),
        ]
        if no_skip:
            cmd.append("--no-skip")
        launched.append(launch_step(cmd, f"{name_prefix}_{shard['device']}"))
        if index < len(devices) - 1 and launch_stagger_seconds > 0:
            time.sleep(float(launch_stagger_seconds))

    workers = []
    for launched_item in launched:
        stdout, _ = launched_item["process"].communicate()
        workers.append(
            {
                "name": launched_item["name"],
                "command": launched_item["command"],
                "return_code": launched_item["process"].returncode,
                "elapsed_seconds": round(time.time() - float(launched_item["started_at"]), 3),
                "stdout_tail": (stdout or "")[-8000:],
            }
        )
    return workers


def parse_cuda_devices(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def load_object_ids(objects_file: Path) -> List[str]:
    payload = load_json(objects_file, default=None)
    if not isinstance(payload, list):
        raise FileNotFoundError(f"Invalid objects file: {objects_file}")
    return [str(item.get("id")).strip() for item in payload if (item or {}).get("id")]


def shard_ids(obj_ids: List[str], devices: List[str]) -> List[dict]:
    buckets = [{"device": device, "obj_ids": []} for device in devices]
    for idx, obj_id in enumerate(obj_ids):
        buckets[idx % len(devices)]["obj_ids"].append(obj_id)
    return [bucket for bucket in buckets if bucket["obj_ids"]]


def copy_meshes(meshes_raw_dir: Path, meshes_dir: Path):
    meshes_dir.mkdir(parents=True, exist_ok=True)
    for glb_path in sorted(meshes_raw_dir.glob("*.glb")):
        shutil.copy2(glb_path, meshes_dir / glb_path.name)


def main():
    args = parse_args()
    objects_file = Path(args.objects_file).resolve()
    output_root = Path(args.output_root).resolve()
    prompts_path = output_root / "prompts.json"
    images_dir = output_root / "images"
    rgba_dir = output_root / "images_rgba"
    meshes_raw_dir = output_root / "meshes_raw"
    meshes_dir = output_root / "meshes"

    output_root.mkdir(parents=True, exist_ok=True)
    obj_ids = load_object_ids(objects_file)
    steps: List[dict] = []

    stage1_cmd = [
        args.python_bin,
        str(STAGE1_SCRIPT),
        "--seed-concepts-file",
        str(objects_file),
        "--output-file",
        str(prompts_path),
    ]
    if args.stage1_mode == "template-only":
        stage1_cmd.append("--template-only")
    elif args.stage1_mode == "dry-run":
        stage1_cmd.append("--dry-run")
    else:
        stage1_cmd.extend(["--model", args.stage1_model])
    steps.append(run_step(stage1_cmd, "stage1_text_expansion"))
    if steps[-1]["return_code"] != 0:
        save_json(output_root / "summary.json", {"success": False, "steps": steps})
        raise SystemExit(1)

    stage2_devices = parse_cuda_devices(args.t2i_device)
    stage2_workers = run_sharded_workers(
        name_prefix="stage2_t2i",
        script_path=STAGE2_SCRIPT,
        python_bin=args.python_bin,
        devices=stage2_devices,
        obj_ids=obj_ids,
        base_args=[
            "--prompts-path",
            str(prompts_path),
            "--output-dir",
            str(images_dir),
            "--seed-base",
            str(args.seed_base),
        ],
        no_skip=args.no_skip,
        launch_stagger_seconds=args.launch_stagger_seconds,
    )
    steps.extend(stage2_workers)
    if not all(step["return_code"] == 0 for step in stage2_workers):
        save_json(output_root / "summary.json", {"success": False, "steps": steps})
        raise SystemExit(1)

    stage25_devices = parse_cuda_devices(args.sam_device)
    stage25_workers = run_sharded_workers(
        name_prefix="stage2_5_sam3_segment",
        script_path=STAGE25_SCRIPT,
        python_bin=args.python_bin,
        devices=stage25_devices,
        obj_ids=obj_ids,
        base_args=[
            "--prompts-path",
            str(prompts_path),
            "--images-dir",
            str(images_dir),
            "--rgba-dir",
            str(rgba_dir),
        ],
        no_skip=args.no_skip,
        launch_stagger_seconds=args.launch_stagger_seconds,
    )
    steps.extend(stage25_workers)
    if not all(step["return_code"] == 0 for step in stage25_workers):
        save_json(output_root / "summary.json", {"success": False, "steps": steps})
        raise SystemExit(1)

    stage3_devices = parse_cuda_devices(args.i23d_devices)
    launched_stage3 = []
    for index, shard in enumerate(shard_ids(obj_ids, stage3_devices)):
        cmd = [
            args.python_bin,
            str(STAGE3_SCRIPT),
            "--device",
            shard["device"],
            "--prompts-path",
            str(prompts_path),
            "--images-dir",
            str(rgba_dir),
            "--rgb-images-dir",
            str(images_dir),
            "--output-dir",
            str(meshes_raw_dir),
            "--ids",
            ",".join(shard["obj_ids"]),
            "--seed-base",
            str(args.seed_base),
        ]
        if args.no_skip:
            cmd.append("--no-skip")
        if args.shape_only:
            cmd.append("--shape-only")
        launched_stage3.append(launch_step(cmd, f"stage3_i23d_{shard['device']}"))
        if index < len(stage3_devices) - 1 and args.launch_stagger_seconds > 0:
            time.sleep(float(args.launch_stagger_seconds))

    stage3_workers = []
    for launched in launched_stage3:
        stdout, _ = launched["process"].communicate()
        stage3_workers.append(
            {
                "name": launched["name"],
                "command": launched["command"],
                "return_code": launched["process"].returncode,
                "elapsed_seconds": round(time.time() - float(launched["started_at"]), 3),
                "stdout_tail": (stdout or "")[-8000:],
            }
        )

    steps.extend(stage3_workers)
    if not all(step["return_code"] == 0 for step in stage3_workers):
        save_json(output_root / "summary.json", {"success": False, "steps": steps})
        raise SystemExit(1)

    copy_meshes(meshes_raw_dir, meshes_dir)
    summary = {
        "success": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "objects_file": str(objects_file),
        "stage1_mode": args.stage1_mode,
        "object_count": len(obj_ids),
        "objects": obj_ids,
        "prompts_path": str(prompts_path),
        "images_dir": str(images_dir),
        "images_rgba_dir": str(rgba_dir),
        "meshes_raw_dir": str(meshes_raw_dir),
        "meshes_dir": str(meshes_dir),
        "steps": steps,
    }
    save_json(output_root / "summary.json", summary)


if __name__ == "__main__":
    main()
