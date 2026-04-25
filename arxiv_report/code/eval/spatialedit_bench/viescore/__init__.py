import sys
sys.path.insert(0, 'viescore')

from utils import (
    mllm_output_to_dict
)
import math
import vie_prompts

class VIEScore:
    def __init__(self, backbone="gpt4o", task="t2i", key_path=None) -> None:
        self.task = task
        self.backbone_name = backbone

        if self.task not in ["t2i", "tie", "t2v"]:
            raise ValueError("task must be either 't2i' or 'tie'")

        if self.backbone_name == "gpt4o":
            from mllm_tools.openai import GPT4o
            self.model = GPT4o(key_path, model_name="gpt-4.1")
        elif self.backbone_name == "gpt4v":
            from mllm_tools.openai import GPT4v
        elif self.backbone_name == "gemini3_pro":
            from mllm_tools.gemini3 import Gemini3pro
            self.model = Gemini3pro(key_path)
        elif self.backbone_name == "gemini3flash":
            from mllm_tools.gemini3 import Gemini3pro
            self.model = Gemini3pro(key_path, model_name="Gemini-3-Flash-Preview")
        elif self.backbone_name == "gemini2flash":
            from mllm_tools.gemini3 import Gemini3pro
            self.model = Gemini3pro(key_path, model_name="Gemini-2.5-flash")
        elif self.backbone_name == "idefics2":
            from mllm_tools.idefics2_eval import Idefics2
            self.model = Idefics2()
        elif self.backbone_name == "mantis":
            from mllm_tools.mantis_idefics2_eval import Mantis
            self.model = Mantis()
        elif self.backbone_name == "minicpmv":
            from mllm_tools.minicpmv_eval import MiniCPMV
            self.model = MiniCPMV()
        elif self.backbone_name == "qwen25vl":
            from mllm_tools.qwen25vl_eval import Qwen25VL
            self.model = Qwen25VL()
        elif self.backbone_name == "qwen35vl":
            from mllm_tools.qwen35vl_eval import Qwen35VL
            self.model = Qwen35VL()
        else:
            raise NotImplementedError("backbone not supported")
        self.context = vie_prompts._context_no_delimit
        if self.task == "t2i":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_image_gen_rule, vie_prompts._prompts_0shot_t2i_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "tie":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ])
        elif self.task == "t2v":
            self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_one_video_gen_rule, vie_prompts._prompts_0shot_t2v_rule_SC])
            self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_t2v_rule_PQ])

    def evaluate(self, image_prompts, prompt, extract_overall_score_only=False, extract_all_score=True, echo_output=False):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if self.backbone_name in ['gpt4o', 'gpt4v', 'gemini3_pro', "gemini2flash", "gemini3flash"]:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
            #print("Using encode:", self.model.use_encode)
        prompt_final = self.model.prepare_prompt(image_prompts, prompt)

        score_dict = False
        tries = 0
        max_tries = 100
        while score_dict is False:
            tries += 1
            guess_if_cannot_parse = True if tries > max_tries else False
            result_score = self.model.get_parsed_output(prompt_final)
            score_dict = mllm_output_to_dict(result_score, give_up_parsing=guess_if_cannot_parse)
            # PQ_dict = {'score': [0]}

        return min(score_dict['score'])

if __name__ == "__main__":
    model = VIEScore(backbone="gemini", task="t2i")
    from datasets import load_dataset
    dataset = load_dataset("TIGER-Lab/GenAI-Arena-Bench", "image_generation")
    dataset = dataset["test"]
    print("Now running the VIEScore model")
    for idx in range(5):
        left_image = dataset['left_image'][idx]
        right_image = dataset['right_image'][idx]
        prompt = dataset['prompt'][idx]
        print(model.evaluate(left_image, prompt, extract_all_score=True))
        print(model.evaluate(right_image, prompt, extract_all_score=True))

