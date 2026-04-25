import os
import torch
from PIL import Image
from typing import List
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Qwen35VL():
    def __init__(self, model_path="/gemini/platform/public/aigc/aigc_image/zhanghy56_intern/zhangqisong/Qwen3-VL-8B-Instruct") -> None:
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.use_encode = True

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        if not isinstance(image_links, list):
            image_links = [image_links]

        content = []
        for img_link in image_links:
            content.append({"type": "image", "image": img_link})
        content.append({"type": "text", "text": "/no_think\n" + text_prompt})

        messages = [{"role": "user", "content": content}]
        return messages

    def get_parsed_output(self, messages):
        set_seed(42)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=1,
            do_sample=False,
            temperature=0.1,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        result = output_text[0] if output_text else ""
        import re as _re
        result = _re.sub(r'<think>.*?</think>', '', result, flags=_re.DOTALL).strip()
        return result


if __name__ == "__main__":
    model = Qwen35VL()
    prompt = model.prepare_prompt(
        ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"],
        "Describe the image in detail.",
    )
    res = model.get_parsed_output(prompt)
    print("result:\n", res)
