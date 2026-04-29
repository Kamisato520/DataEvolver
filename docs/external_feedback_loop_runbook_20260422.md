# External Feedback Loop Runbook

This runbook describes the first usable outer loop for ARIS scene rotation data:

```text
Stage1->dataset generation
-> training
-> eval/bench metrics
-> external feedback analysis
-> feedback-augmented dataset
-> optional next training run
```

## Main Script

```bash
/home/jiazhuangzhuang/ARIS/scripts/run_stage1_to_externalfb_train_loop.sh
```

The script is intentionally conservative:

- It does not overwrite existing dataset roots.
- It keeps training and eval commands pluggable through environment variables.
- It writes logs and state under `runtime/external_feedback_loop/<round>/`.
- It exports `DATASET_ROOT` and `TRAIN_OUTPUT_DIR` before running the training command.

## Minimal Existing-Dataset Loop

Use this when the dataset already exists and you only want to train, analyze eval metrics, and create the next feedback dataset.

```bash
cd /home/jiazhuangzhuang/ARIS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aris

TRAIN_CMD='bash /aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.sh' \
PYTHON_BIN=python \
bash scripts/run_stage1_to_externalfb_train_loop.sh \
  --round externalfb_r01_20260422 \
  --dataset-root pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410 \
  --eval-zip /path/to/eval_spatialedit_metrics.zip
```

## Full Stage1-To-Training Loop

Use this when you want to regenerate assets/data from Stage1 before training.

```bash
cd /home/jiazhuangzhuang/ARIS
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aris

TRAIN_CMD='bash /aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.sh' \
BLENDER_BIN=/aaaidata/zhangqisong/blender-4.24/blender \
PYTHON_BIN=python \
PYTHONPATH=/aaaidata/zhangqisong/DiffSynth-Studio:${PYTHONPATH:-} \
bash scripts/run_stage1_to_externalfb_train_loop.sh \
  --round stage1_externalfb_r01_20260422 \
  --run-stage1 \
  --dataset-root pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410 \
  --eval-zip /path/to/eval_spatialedit_metrics.zip
```

If the Stage1 generation command needs to be customized, pass it through `STAGE1_DATASET_CMD`.

## Train Again On Feedback Dataset

Add `--train-feedback` to run the training command a second time on the feedback-augmented dataset:

```bash
TRAIN_CMD='bash /aaaidata/zhangqisong/DiffSynth-Studio/train_clockwise.sh' \
bash scripts/run_stage1_to_externalfb_train_loop.sh \
  --round externalfb_r01_20260422 \
  --dataset-root pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_final_20260410 \
  --eval-zip /path/to/eval_spatialedit_metrics.zip \
  --train-feedback
```

The second training run receives:

```bash
DATASET_ROOT=<feedback dataset root>
TRAIN_OUTPUT_DIR=runtime/external_feedback_loop/<round>/train_feedback
```

## Outputs

For each round:

```text
runtime/external_feedback_loop/<round>/
  LOOP_STATE.json
  logs/
  feedback/
    feedback_summary.json
    augmentation_requirements.json
    weak_samples.csv
    DATASET_FEEDBACK_PLAN.md
  train_first/
  train_feedback/
```

The generated feedback dataset root defaults to:

```text
pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_externalfb_<round>
```

## Current Verified Feedback Dataset

The first feedback dataset generated on 2026-04-22 is:

```text
pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_externalfb_angle050607_20260422
```

Counts:

```text
train: 350
val: 49
test: 56
all: 455
external_feedback: 105
```
