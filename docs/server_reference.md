# 服务器环境参考

## 68 服务器（训练/评测主力，8×H100）

- SSH: `zhanghy56_68`
- GPU: 8×H100
- 工作目录: `/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/data-build`（下文简称 `$WORKDIR`）
- **Python 环境**: `source .venv/bin/activate`（uv 环境，在工作目录下）
- 依赖安装：`uv pip install`（不是 pip install）
- Qwen-Image-Edit-2511 模型: `/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen-Image-Edit-2511`
- accelerate 配置: `DiffSynth-Studio/accelerate_config_6gpu.yaml`
- Blender 路径：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/transfer_part_2/blender/blender-4.2.4-linux-x64`
- fal LoRA: **未传输**（需从 wwz 手动传到 `$WORKDIR/data/fal_lora/`）

### 已部署脚本（`DiffSynth-Studio/` 下）

| 脚本 | 用途 |
|------|------|
| `train_clockwise.py` / `.sh` | LoRA 训练 |
| `eval_inference.py` | 推理（base/ours/fal，自动检查 LoRA 是否存在） |
| `eval_metrics.py` | 指标计算（PSNR/SSIM/LPIPS/CLIP-I/DINO/FID） |
| `run_eval_inference.sh` | 推理 wrapper（自动跳过缺失 LoRA） |
| `run_eval_metrics.sh` | 指标 wrapper（自动安装缺失依赖） |
| `run_full_eval.sh` | 一键全流程：推理 → 指标 |
| `eval_spatialedit_inference.py` | SpatialEdit-Bench 推理（base/ours/fal，61 obj × 8 angles） |
| `eval_spatialedit_metrics.py` | SpatialEdit-Bench 指标（PSNR/SSIM/LPIPS/CLIP-I/DINO） |
| `prepare_spatialedit_folders.py` | 为 eval_image_metrics.py 准备 pred/gt symlink 文件夹 |
| `run_spatialedit_eval.sh` | SpatialEdit-Bench 评测（3 mode 并行，GPU 0/2/4） |
| `run_objinfo_full_eval.sh` | Objinfo LoRA 推理+评测一体化流水线（断点续传） |
| `run_spatialedit_viescore_sequential.sh` | SpatialEdit-Bench VIEScore 评测（顺序，单 GPU） |
| `run_spatialedit_viescore_parallel.sh` | SpatialEdit-Bench VIEScore 评测（4 mode 并行，4 GPU） |

### 常用命令

```bash
# 检查 GPU
ssh zhanghy56_68 "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
# 查训练进度
ssh zhanghy56_68 "tmux capture-pane -t bbox-train -p | tail -5"
# 训练完成后一键评测
ssh zhanghy56_68 "cd $WORKDIR/DiffSynth-Studio && CUDA_VISIBLE_DEVICES=0 bash run_full_eval.sh 30"
```

### 评测结果路径

- Test Set 指标：`$WORKDIR/DiffSynth-Studio/output/eval_metrics/`
- SpatialEdit-Bench 传统指标 CSV：`$WORKDIR/DiffSynth-Studio/output/eval_spatialedit_metrics/{mode}_metrics.csv`
- SpatialEdit-Bench VIEScore CSV：`$WORKDIR/SpatialEdit-Bench-Eval/csv_results/{mode}/qwen35vl/{mode}_rotate_en_vie_score.csv`
- VLM backbone：`/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen3-VL-8B-Instruct`
- 评测代码：`$WORKDIR/SpatialEdit-Bench-Eval/object_level_eval/`

---

## intern 服务器（备用，与 68 共享磁盘）

- SSH: `zhanghy56_intern`
- GPU: 8×H100
- 与 68 服务器**共享磁盘**，工作目录和路径完全相同
- **后续不再主动使用**，所有训练/评测统一在 68 上执行
- 已有 checkpoint（如 `rotation8_bright_objinfo_rank32/epoch_0029`）可直接在 68 上读取

---

## wwz 服务器（数据构建）

- SSH: `wwz`（密钥免密）
- GPU: 3×A800 80GB
- **Python 环境**（训练用）：`/home/wuwenzhuo/Qwen-VL-Series-Finetune/env/bin/python3`（不用 conda activate）
- Blender binary: `/home/wuwenzhuo/blender-4.24/blender`
- 场景文件: `/home/wuwenzhuo/blender/data/sence/4.blend`（注意是 `sence` 非 `scene`）
- Qwen3.5-35B 模型: `/data/wuwenzhuo/Qwen3.5-35B-A3B`
- Qwen-Image-Edit-2511 模型: `/data/wuwenzhuo/Qwen-Image-Edit-2511`
- fal 社区 LoRA: `/data/wuwenzhuo/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/qwen-image-edit-2511-multiple-angles-lora.safetensors`
- DiffSynth-Studio: `/aaaidata/zhangqisong/DiffSynth-Studio`（uv 环境，训练用 Python 全路径）
- accelerate 配置: `DiffSynth-Studio/accelerate_config_3gpu.yaml`（MULTI_GPU bf16）
- **必须用 tmux**（screen 不可用）

```bash
# 检查 GPU
ssh wwz "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"
# 查训练进度
ssh wwz "tmux capture-pane -t bright-train -p | tail -20"
```

---

## LoRA key 格式转换

- 我们的 checkpoint: 去掉 `pipe.dit.` 前缀和 `.default.`
- fal 社区 LoRA: 去掉 `transformer.` 前缀
- DiffSynth 内部格式: `transformer_blocks.0.attn.to_q.lora_A.weight`
