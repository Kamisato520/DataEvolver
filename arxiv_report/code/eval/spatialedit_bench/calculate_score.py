from viescore import VIEScore
import PIL
import os
import os
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import sys
import csv
import time
import argparse
import math
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import copy
from collections import defaultdict
from PIL import Image
try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    build_sam3_image_model = None
    Sam3Processor = None
from viescore import vie_prompts


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)


def calculate_iou_score(sam3_processor, item, edit_image_path):
    """
    box format: [x1, y1, x2, y2]
    """
    image_pil = Image.open(edit_image_path)
    inference_state_state = sam3_processor.set_image(image_pil)
    seg_prompt = item['object']
    box1 = sam3_processor.set_text_prompt(state=inference_state_state, prompt=seg_prompt)["boxes"]
    if len(box1) == 0:
        return 0.0
    box1 = box1[0].tolist()

    box2 = [float(x) for x in json.loads(item["bbox_gt"])]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = box1_area + box2_area - inter_area + 1e-6

    return inter_area / union

def extract_view_from_instruction(instruction):
    for k, v in vie_prompts.VIEW_PROMPT.items():
        if v in instruction:
            return k
    return None

def process_single_item(item, vie_score, edit_task, sam3_processor=None, bench_data_dir="", max_retries=10000):

    instruction = item['instruction']
    key = item['key']
    instruction_language = item['instruction_language']
    edit_image_path = item['edited_image_path']

    if edit_task == 'move':
        iou_score = calculate_iou_score(sam3_processor, item, edit_image_path)

    for retry in range(max_retries):
        try:
            pil_image = Image.open(os.path.join(bench_data_dir, item['image_path'])).convert("RGB")
            pil_image_edited = Image.open(edit_image_path).convert("RGB")
            source_img_width, source_img_height, _ = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
            edited_img_width, edited_img_height, _ = calculate_dimensions(512 * 512, pil_image_edited.width / pil_image_edited.height)
            pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
            pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))

            context = vie_prompts._context_no_delimit
            image_prompts = [pil_image, pil_image_edited]
            if edit_task == 'move':
                Score_oc_prompt = "\n".join([context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC_move])
                Score_oc_prompt = Score_oc_prompt.replace("<instruction>", instruction)
                Score_oc = vie_score.evaluate(image_prompts, Score_oc_prompt) / 10
                overall_score = math.sqrt(Score_oc * iou_score)
            elif edit_task == "rotate":
                num_id = extract_view_from_instruction(instruction)
                question_prompt = vie_prompts.SC_rotate[num_id].format(object_name=item["object"])
                Score_view_prompt = "\n".join([vie_prompts._prompts_0shot_tie_rule_SC_rotate, question_prompt])
                Score_cons_prompt = "\n".join([context, vie_prompts._prompts_0shot_in_context_generation_rule_SC_Scene])
                Score_cons_prompt = Score_cons_prompt.replace("<instruction>", instruction)
                Score_view = vie_score.evaluate(image_prompts[-1:], Score_view_prompt) / 10
                Score_cons = vie_score.evaluate(image_prompts, Score_cons_prompt) / 10
                overall_score = math.sqrt(Score_view * Score_cons)

            if edit_task == 'move':
                print(f"Score_oc: {Score_oc}, iou_score: {iou_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
                return {
                    "key": key,
                    "edited_image": edit_image_path,
                    "instruction": instruction,
                    "Score_oc": Score_oc,
                    "iou_score": iou_score,
                    "instruction_language" : item['instruction_language']
                }
            elif edit_task == "rotate":
                print(f"Score_view: {Score_view}, Score_cons: {Score_cons}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}")
                return {
                    "key": key,
                    "edited_image": edit_image_path,
                    "instruction": instruction,
                    "Score_view": Score_view,
                    "Score_cons": Score_cons,
                    "instruction_language" : item['instruction_language']
                }
            else:
                raise

        except Exception as e:
            if retry < max_retries - 1:
                wait_time = (retry + 1) * 2  # 指数退避：2秒, 4秒, 6秒...
                print(f"Error processing (attempt {retry + 1}/{max_retries}): {e}")
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to process {save_path_item} after {max_retries} attempts: {e}")
                return

def find_files_with_given_basename(folder_path, basename):
    pattern = os.path.join(folder_path, f"{basename}.*")
    matched_files = glob.glob(pattern)
    return [os.path.basename(f) for f in matched_files]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="spatialedit", help="edit model name")
    parser.add_argument("--edited_images_dir", type=str, default="edit_images", help="path to edited images")
    parser.add_argument("--instruction_language", type=str, default="en", choices=["all", "en"])
    parser.add_argument("--sam3_ckpt", type=str, default="/pfs/yichengxiao/huggingface/model/sam3/sam3.pt")
    parser.add_argument("--metadata_path", type=str, default="/pfs/yichengxiao/data/space_edit/SpatialEdit_Bench_Meta_File.json")
    parser.add_argument("--bench-data-dir", type=str, default="")
    parser.add_argument("--type", type=str, default="all",  choices=["all", "move", "rotate"])
    parser.add_argument("--save_dir", type=str, default="csv_results")
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["gpt4o", "qwen25vl", "qwen35vl", "gemini3_pro", "gemini2flash", "gemini3flash"])
    args = parser.parse_args()
    model_name = args.model_name
    edited_images_dir = args.edited_images_dir
    instruction_language = args.instruction_language
    save_dir = args.save_dir
    backbone = args.backbone
    process_language = ['en']
    if args.type == "all":
        groups = ["move", "rotate"]
        # groups = ["move"]
        # groups = ["rotate_gt"]
    else:
        groups = [args.type]

    # Load GEdit-Bench dataset and group by task type
    vie_score = VIEScore(backbone=backbone, task="tie", key_path='secret.env')
    sam3_processor = None
    if args.type != "rotate":
        model_meta = build_sam3_image_model(checkpoint_path=args.sam3_ckpt)
        sam3_processor = Sam3Processor(model_meta)

    metadata_path = args.metadata_path
    with open(metadata_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset_by_group = defaultdict(list)
    for i, item in tqdm(enumerate(dataset), desc="Loading space-Bench dataset..."):
        dataset_by_group[item['type']].append(item)
    for k, v in dataset_by_group.items():
        print(f"Number of samples in {k} - {instruction_language}:", len(v))

    # Evaluate each group
    save_path_new = os.path.join(save_dir, model_name, backbone)
    for group_name in groups:
        group_csv_list = []
        group_dataset_list = dataset_by_group[group_name]

        # Load existing group CSV if it exists, if csv esists, skip this group
        group_csv_path = os.path.join(save_path_new, f"{model_name}_{group_name}_{instruction_language}_vie_score.csv")
        if os.path.exists(group_csv_path):
            with open(group_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                group_results = list(reader)
            print(f"{model_name} - {group_name} exsits, skip this group")
            continue
        
        if backbone in ["gpt4o", "gemini3_pro", "gemini2flash", "gemini3flash"]:
            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = []
                for item in group_dataset_list:
                    for language in process_language:
                        temp_item = copy.deepcopy(item)
                        if language == 'cn':
                            temp_item['instruction'] = temp_item['instruction_zh']
                            temp_item['instruction_language'] = 'cn'
                        else:
                            temp_item['instruction_language'] = 'en'
                        key = item["image_id"]
                        temp_item['key'] = key
                        temp_item['instruction'] = item['prompt']
                        try:
                            # Should organize edited image directory, please refer EVAL.md for details
                            edited_images_path = os.path.join(edited_images_dir, model_name, 'fullset', group_name, language)
                            temp_item['edited_image_path'] = os.path.join(edited_images_path, '/'.join([temp_item['image_id'], temp_item['edit_id'] + '.png']))
                        except:
                            print(key, "not found in", edited_images_path)
                            continue

                        # Check if this sample has already been processed
                        future = executor.submit(process_single_item, temp_item, vie_score, group_name, sam3_processor, args.bench_data_dir)
                        futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {model_name} - {group_name}"):
                    result = future.result()
                    if result:
                        group_csv_list.append(result)

        else:
            for item in tqdm(group_dataset_list, desc=f"Processing {model_name} - {group_name}"):
                for language in process_language:
                    temp_item = copy.deepcopy(item)
                    temp_item['instruction_language'] = language
                    key = item["image_id"]
                    temp_item['key'] = key
                    temp_item['instruction'] = item['prompt']
                    try:
                        edited_images_path = os.path.join(edited_images_dir, model_name, 'fullset', group_name, language)
                        temp_item['edited_image_path'] = os.path.join(edited_images_path, '/'.join([temp_item['image_id'], temp_item['edit_id'] + '.png']))
                    except:
                        print(key, "not found in", edited_images_path)
                        continue

                    result = process_single_item(temp_item, vie_score, group_name, sam3_processor, args.bench_data_dir)
                    if result:
                        group_csv_list.append(result)

        # Save group-specific CSV
        if group_name == "move":
            with open(group_csv_path, 'w', newline='') as f:
                fieldnames = ["key", "edited_image", "instruction", "Score_oc", "iou_score", "instruction_language"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in group_csv_list:
                    writer.writerow(row)
            print(f"Saved group CSV for {group_name}, length： {len(group_csv_list)}")
        elif group_name == "rotate":
            with open(group_csv_path, 'w', newline='') as f:
                fieldnames = ["key", "edited_image", "instruction", "Score_view", "Score_cons", "instruction_language"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in group_csv_list:  
                    writer.writerow(row)
            print(f"Saved group CSV for {group_name}, length： {len(group_csv_list)}")
