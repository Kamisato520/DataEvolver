import os
import os
import pandas as pd
from collections import defaultdict
import sys
import numpy as np
import math
from contextlib import redirect_stdout


GROUPS = [
    "move", "rotate"
    # "move"
    # "rotate_gt"
]


def analyze_scores(save_path_dir, evaluate_group, language, file_ext):
    group_scores_view = defaultdict(lambda: defaultdict(list))
    group_scores_cons = defaultdict(lambda: defaultdict(list))
    group_scores_oc = defaultdict(lambda: defaultdict(list))
    group_scores_iou = defaultdict(lambda: defaultdict(list))
    group_scores_move_overall = defaultdict(lambda: defaultdict(list))
    group_scores_rotate_overall = defaultdict(lambda: defaultdict(list))
    
    for group_name in GROUPS:
        csv_path = os.path.join(save_path_dir, f"{evaluate_group[0]}_{group_name}_{file_ext}_vie_score.csv")
        csv_file = open(csv_path)
        df = pd.read_csv(csv_file)
        
        if group_name == 'move':
            filtered_oc_scores = []
            filtered_iou_scores = []
        elif group_name == "rotate":
            filtered_cons_scores = []
            filtered_view_scores = []
        else:
            raise
        filtered_overall_scores = []

        for _, row in df.iterrows():
            instruction_language = row['instruction_language']
            if instruction_language == language:
                pass
            else:
                continue

            if group_name == 'move':
                Score_oc = row['Score_oc']
                iou_score = row['iou_score']
                overall_score = math.sqrt(Score_oc * iou_score)
                filtered_iou_scores.append(iou_score)
                filtered_oc_scores.append(Score_oc)
            elif group_name == "rotate":
                Score_view = row['Score_view']
                Score_cons = row['Score_cons']
                overall_score = math.sqrt(Score_view * Score_cons)
                filtered_cons_scores.append(Score_cons)
                filtered_view_scores.append(Score_view)
            
            filtered_overall_scores.append(overall_score)
        
        if group_name == 'move':
            avg_oc_score = np.mean(filtered_oc_scores)
            avg_iou_score = np.mean(filtered_iou_scores)
            group_scores_oc[evaluate_group[0]][group_name] = avg_oc_score
            group_scores_iou[evaluate_group[0]][group_name] = avg_iou_score
            avg_overall_score = np.mean(filtered_overall_scores)
            group_scores_move_overall[evaluate_group[0]][group_name] = avg_overall_score
        elif group_name == 'rotate':
            avg_cons_score = np.mean(filtered_cons_scores)
            avg_view_score = np.mean(filtered_view_scores)
            group_scores_cons[evaluate_group[0]][group_name] = avg_cons_score
            group_scores_view[evaluate_group[0]][group_name] = avg_view_score
            avg_overall_score = np.mean(filtered_overall_scores)
            group_scores_rotate_overall[evaluate_group[0]][group_name] = avg_overall_score
        else:
            raise

    # --- Overall Model Averages ---

    if "move" in GROUPS:
        # score oc
        for model_name in evaluate_group:
            model_scores = [group_scores_oc[model_name]["move"]]
            model_avg = np.mean(model_scores)
            group_scores_oc[model_name]["avg_score_oc"] = model_avg
        # score IoU
        for model_name in evaluate_group:
            model_scores = [group_scores_iou[model_name]["move"]]
            model_avg = np.mean(model_scores)
            group_scores_iou[model_name]["avg_score_iou"] = model_avg
        # score overall
        for model_name in evaluate_group:
            model_scores = [group_scores_move_overall[model_name]["move"]]
            model_avg = np.mean(model_scores)
            group_scores_move_overall[model_name]["avg_overall"] = model_avg
    else:
        for model_name in evaluate_group:
            group_scores_oc[model_name]["avg_score_oc"] = 0
            group_scores_iou[model_name]["avg_score_iou"] = 0
            group_scores_move_overall[model_name]["avg_overall"] = 0

    if "rotate" in GROUPS:
        # score cons
        for model_name in evaluate_group:
            model_scores = [group_scores_cons[model_name]["rotate"]]
            model_avg = np.mean(model_scores)
            group_scores_cons[model_name]["avg_score_cons"] = model_avg
        # score view
        for model_name in evaluate_group:
            model_scores = [group_scores_view[model_name]["rotate"]]
            model_avg = np.mean(model_scores)
            group_scores_view[model_name]["avg_score_view"] = model_avg
        # score overall
        for model_name in evaluate_group:
            model_scores = [group_scores_rotate_overall[model_name]["rotate"]]
            model_avg = np.mean(model_scores)
            group_scores_rotate_overall[model_name]["avg_overall"] = model_avg
    else:
        for model_name in evaluate_group:
            group_scores_cons[model_name]["avg_score_cons"] = 0
            group_scores_view[model_name]["avg_score_view"] = 0
            group_scores_rotate_overall[model_name]["avg_overall"] = 0

    return group_scores_oc, group_scores_iou, group_scores_cons, group_scores_view, group_scores_move_overall, group_scores_rotate_overall

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="step1x")
    parser.add_argument("--backbone", type=str, default="gpt4o", choices=["gpt4o", "qwen25vl", "qwen35vl", "gemini3_pro", "gemini2flash", "gemini3flash"])
    parser.add_argument("--save_path", type=str, default="/results/")
    parser.add_argument("--language", type=str, default="all", choices=["all", "en", "cn"])
    args = parser.parse_args()
    model_name = args.model_name
    backbone = args.backbone
    save_path_dir = args.save_path
    if args.language == "all":
        languages = ["en", "cn"]
    else:
        languages = [args.language]

    save_path_txt = f"{backbone}_results.txt"
    final_result_save_path = os.path.join(save_path_dir, model_name)
    os.makedirs(final_result_save_path, exist_ok=True)

    with open(os.path.join(final_result_save_path, save_path_txt), "w") as f:
        with redirect_stdout(f):  # 重定向所有 print 到文件
            for language in languages:
                print("="*10 + f" backbone:{backbone} - model_name:{model_name} - language:{language} " + "="*10)

                save_path_new = os.path.join(save_path_dir, model_name, backbone)
                group_scores_oc, group_scores_iou, group_scores_cons, group_scores_view, group_scores_move_overall, group_scores_rotate_overall = analyze_scores(save_path_new, [model_name], language=language, file_ext=args.language)

                print("\nOverall:")
                for group_name in GROUPS:
                    if group_name == "move":
                        print(f"{group_name}: scores_oc score_iou {group_name}_overall")
                        print(f"{group_name}: {group_scores_oc[model_name][group_name]:.3f}, "
                              f"{group_scores_iou[model_name][group_name]:.3f}, "
                              f"{group_scores_move_overall[model_name][group_name]:.3f}")
                        print(f"Average: {group_scores_oc[model_name]['avg_score_oc']:.3f}, "
                              f"{group_scores_iou[model_name]['avg_score_iou']:.3f}, "
                              f"{group_scores_move_overall[model_name]['avg_overall']:.3f}")

                    else:
                        print(f"{group_name}: scores_cons score_view {group_name}_overall")
                        print(f"{group_name}: {group_scores_cons[model_name][group_name]:.3f}, "
                              f"{group_scores_view[model_name][group_name]:.3f}, "
                              f"{group_scores_rotate_overall[model_name][group_name]:.3f}")
                        print(f"Average: {group_scores_cons[model_name]['avg_score_cons']:.3f}, "
                              f"{group_scores_view[model_name]['avg_score_view']:.3f}, "
                              f"{group_scores_rotate_overall[model_name]['avg_overall']:.3f}")
