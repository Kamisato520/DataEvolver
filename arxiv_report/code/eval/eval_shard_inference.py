"""Shard-based inference for 6-GPU parallel eval."""
import sys, os, argparse
WORKDIR = "/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build"
sys.path.insert(0, f"{WORKDIR}/DiffSynth-Studio")
from eval_spatialedit_inference import load_pipeline, load_lora_into_pipe, collect_pairs, run_inference

parser = argparse.ArgumentParser()
parser.add_argument("--shard-id", type=int, required=True)
parser.add_argument("--num-shards", type=int, required=True)
parser.add_argument("--lora-path", required=True)
parser.add_argument("--output-dir", required=True)
args = parser.parse_args()

pairs = collect_pairs()
obj_names = sorted(set(p["obj_name"] for p in pairs))
shard_objs = set(obj_names[args.shard_id::args.num_shards])
shard_pairs = [p for p in pairs if p["obj_name"] in shard_objs]
print(f"Shard {args.shard_id}/{args.num_shards}: {len(shard_pairs)} pairs ({len(shard_objs)} objects)")

pipe = load_pipeline(device="cuda:0")
load_lora_into_pipe(pipe, args.lora_path, mode="ours")
run_inference(pipe, shard_pairs, args.output_dir)
