# Full50 Stage1 Expansion Runbook

## Goal

Expand the validated `full20` scene dataset pipeline to `full50` without mutating any existing `full20` dataset roots.

For `obj_021` to `obj_050`, the pipeline now starts **from Stage 1**:

1. `stage1_text_expansion`
2. `stage2_t2i_generate`
3. `stage2_5_sam2_segment`
4. `stage3_image_to_3d`
5. `yaw000` bootstrap only
6. consistent `rotation4` / `rotation8` export
7. merge with frozen `full20`
8. build `train-ready` / `split` / `geommeta` / `geomodal`

## Frozen Roots

These roots are treated as read-only inputs:

- `pipeline/data/dataset_scene_v7_full20_rotation4_consistent_yaw000_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_consistent_yaw000_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_trainready_front2others_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_trainready_front2others_splitobj_seed42_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_geommeta_from_consistent_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_20260407`
- `pipeline/data/dataset_scene_v7_full20_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260407`

## New Object Definitions

Seed concept files:

- [scene_obj021_050_objects.json](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/configs/seed_concepts/scene_obj021_050_objects.json)
- [scene_full50_objects.json](/D:/新建文件夹/用户目录/桌面/autoresearch_vlm/Auto-claude-code-research-in-sleep/DataEvolver/DataEvolver/configs/seed_concepts/scene_full50_objects.json)

The new 30 objects are:

- `obj_021 folding_chair`
- `obj_022 side_table`
- `obj_023 electric_kettle`
- `obj_024 table_fan`
- `obj_025 camping_lantern`
- `obj_026 storage_crate`
- `obj_027 hand_truck`
- `obj_028 wheelbarrow`
- `obj_029 step_ladder`
- `obj_030 picnic_cooler`
- `obj_031 pickup_truck`
- `obj_032 scooter`
- `obj_033 delivery_van`
- `obj_034 golf_cart`
- `obj_035 traffic_barrier`
- `obj_036 road_bollard`
- `obj_037 parking_meter`
- `obj_038 street_sign`
- `obj_039 planter_box`
- `obj_040 street_trash_can`
- `obj_041 basketball`
- `obj_042 soccer_ball`
- `obj_043 tennis_racket`
- `obj_044 baseball_bat`
- `obj_045 surfboard`
- `obj_046 life_ring`
- `obj_047 barbecue_grill`
- `obj_048 portable_generator`
- `obj_049 shopping_cart`
- `obj_050 fire_extinguisher`

## New Output Roots

Stage1-3 sandbox root:

- `pipeline/data/dataset_scene_v7_obj021_obj050_stage1_assets_20260408`

Yaw000 bootstrap root:

- `pipeline/data/evolution_scene_v7_obj021_obj050_yaw000_bootstrap_20260408`

New 30-object consistent roots:

- `pipeline/data/dataset_scene_v7_obj021_obj050_rotation4_consistent_yaw000_20260408`
- `pipeline/data/dataset_scene_v7_obj021_obj050_rotation8_consistent_yaw000_20260408`

Merged full50 roots:

- `pipeline/data/dataset_scene_v7_full50_rotation4_consistent_yaw000_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_consistent_yaw000_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_trainready_front2others_splitobj_seed42_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geommeta_from_consistent_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_20260408`
- `pipeline/data/dataset_scene_v7_full50_rotation8_geomodal_trainready_front2others_splitobj_seed42_20260408`

## Validation Status

Completed locally:

- metadata precheck passes for `obj_021` to `obj_050`
- `stage1_text_expansion --template-only` successfully generated prompts for all 30 new objects
- `bootstrap_scene_yaw000_objects.py --help` passes
- `build_scene_full50_expansion_pipeline.py --help` passes
- all modified scripts pass `py_compile`

Not completed locally:

- Stage 2 / Stage 2.5 / Stage 3 execution
- Blender scene bootstrap and export
- full50 dataset materialization

These still require the actual model/runtime environment.

## Commands

### 1. Metadata-only precheck

```bash
python scripts/precheck_scene_full50_assets.py \
  --objects-file configs/seed_concepts/scene_obj021_050_objects.json \
  --full-objects-file configs/seed_concepts/scene_full50_objects.json
```

### 2. Stage1 prompt smoke test

```bash
python pipeline/stage1_text_expansion.py \
  --seed-concepts-file configs/seed_concepts/scene_obj021_050_objects.json \
  --output-file _tmp/scene_obj021_050_prompts_test_20260408.json \
  --template-only
```

### 3. Full pipeline launch

Replace `<python>` and `<blender>` with the target runtime paths.

```bash
python scripts/build_scene_full50_expansion_pipeline.py \
  --python <python> \
  --blender <blender> \
  --stage1-mode template-only \
  --t2i-device cuda:0 \
  --sam-device cuda:0 \
  --i23d-devices cuda:0,cuda:1,cuda:2 \
  --bootstrap-gpus 0,1,2 \
  --export-gpus 0,1,2 \
  --asset-mode symlink \
  --train-objects 35 \
  --val-objects 7 \
  --test-objects 8 \
  --seed 42
```

## Notes

- The new 30 objects do **not** reuse the old `pipeline/data/meshes` as their canonical source.
- The canonical source for `obj_021` to `obj_050` is the isolated Stage1-3 sandbox root.
- `rotation4` and `rotation8` must always be exported from the chosen `yaw000` best state.
- The old `full20` watchdog/tuner scripts remain untouched and are not part of the `full50` launch path.
