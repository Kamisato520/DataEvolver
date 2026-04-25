# This file is generated automatically through parse_prompt.py

_context_no_delimit = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (Keep your reasoning concise and short.):
{
"score" : [...],
"reasoning" : "..."
}"""

_context = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.

You will have to give your output in this way (the delimiter is necessary. Keep your reasoning concise and short.):
||V^=^V||
{
"score" : 
"reasoning" : 
}
||V^=^V||"""

_context_no_format = """You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.
All the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials."""

_prompts_1shot_multi_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: A digital illustration of a cat beside a wooden pot
Output:
||V^=^V||
{
"score" : [5, 5, 10],
"reasoning" :  "The cat is not beside the wooden pot. The pot looks partially resemble to the subject pot. The cat looks highly resemble to the subject cat."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>"""

_prompts_1shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

First lets look at the first set of input (1st image) as an example. 
Text Prompt: A pink and a white frisbee are on the ground.
Output:
||V^=^V||
{
"score" : [5],
"reasoning" :  "White frisbee not present in the image."
}
||V^=^V||

Now evaluate the second set of input (2nd image).
Text Prompt: <prompt>
"""

_prompts_1shot_tie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st and 2nd images) as an example. 
Editing instruction: What if the man had a hat?
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The hat exists but does not suit well. The hat also looks distorted. But it is a good edit because only a hat is added and the background is persevered."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Editing instruction: <instruction>
"""

_prompts_1shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

First lets look at the first set of input (1st, 2nd and 3rd images) as an example. 
Subject: <subject>
Output:
||V^=^V||
{
"score" : [5, 10],
"reasoning" :  "The monster toy looks partially resemble to the token subject. The edit is minimal."
}
||V^=^V||

Now evaluate the second set of input (4th, 5th, and 6th images).
Subject: <subject>
"""

_prompts_1shot_one_image_gen_rule = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: a red cartoon figure eating a banana
Output:
||V^=^V||
{
"score" : [10, 5],
"reasoning" :  "The red cartoon figure is eating a banana. The red cartoon figure looks partially resemble to the subject."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_rule_PQ = """RULES of each set of inputs:

One image will be provided; The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]


First lets look at the first set of input (1st image) as an example. 
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The image gives an unnatural feeling on hands of the girl. There is also minor distortion on the eyes of the girl."
}
||V^=^V||

Now evaluate the second set of input (2nd image).

"""

_prompts_1shot_subject_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_1shot_cig_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

First lets look at the first set of input (1st and 2nd images) as an example. 
Text Prompt: the bridge is red, Golden Gate Bridge in San Francisco, USA
Output:
||V^=^V||
{
"score" : [5, 5],
"reasoning" :  "The bridge is red. But half of the bridge is gone."
}
||V^=^V||

Now evaluate the second set of input (3th, 4th images).
Text Prompt: <prompt>
"""

_prompts_1shot_two_image_edit_rule = """RULES of each set of inputs:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_1shot_subject_image_edit_rule = """RULES of each set of inputs:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_1shot_control_image_gen_rule = """RULES of each set of inputs:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_two_image_edit_rule = """RULES:

Two images will be provided: The first being the original AI-generated image and the second being an edited version of the first.
The objective is to evaluate how successfully the editing instruction has been executed in the second image.

Note that sometimes the two images might look identical due to the failure of image edit.
"""

_prompts_0shot_one_video_gen_rule = """RULES:

The images are extracted from a AI-generated video according to the text prompt.
The objective is to evaluate how successfully the video has been generated.
"""

_prompts_0shot_t2v_rule_PQ = """RULES:

The image frames are AI-generated.
The objective is to evaluate how successfully the image frames has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on the image frames naturalness. 
(
    0 indicates that the scene in the image frames does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image frames looks natural.
)
A second score from 0 to 10 will rate the image frames artifacts. 
(
    0 indicates that the image frames contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image frames has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_msdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the first sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the first sub-image.)
A third score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second sub-image. 
(0 indicates that the subject in the second image does not look like the token subject in the second sub-image at all. 10 indicates the subject in the second image look exactly alike the token subject in the second sub-image.)
Put the score in a list such that output score = [score1, score2, score3], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance for the first sub-image, and 'score3' evaluates the resemblance for the second sub-image.

Text Prompt: <prompt>
"""

_prompts_0shot_sdie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the second image. 
(0 indicates that the subject in the third image does not look like the token subject at all. 10 indicates the subject in the third image look exactly alike the token subject.)
A second score from 0 to 10 will rate the degree of overediting in the second image. 
(0 indicates that the scene in the edited image is completely different from the first image. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the resemblance and 'score2' evaluates the degree of overediting.

Subject: <subject>"""

_prompts_0shot_subject_image_edit_rule = """RULES:

Three images will be provided: 
The first image is a input image to be edited.
The second image is a token subject image.
The third image is an AI-edited image from the first image. it should contain a subject that looks alike the subject in second image.
The objective is to evaluate how successfully the image has been edited.
"""

_prompts_0shot_mie_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_sdig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the subject in the generated image resemble to the token subject in the first image. 
(0 indicates that the subject in the second image does not look like the token subject at all. 10 indicates the subject in the second image look exactly alike the token subject.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the resemblance.

Text Prompt: <prompt>
"""

_prompts_0shot_tie_rule_SC = """
From scale 0 to 10: 
A score from 0 to 10 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 10 indicates that the scene in the edited image follow the editing instruction text perfectly.)
A second score from 0 to 10 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 10 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.

Editing instruction: <instruction>
"""

_prompts_0shot_t2i_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the AI generated image does not follow the prompt at all. 10 indicates the AI generated image follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""

_prompts_0shot_cig_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the second image does not follow the prompt at all. 10 indicates the second image follows the prompt perfectly.)
A second score from 0 to 10 will rate how well the generated image is following the guidance image. 
(0 indicates that the second image is not following the guidance at all. 10 indicates that second image is following the guidance image.)
Put the score in a list such that output score = [score1, score2], where 'score1' evaluates the prompt and 'score2' evaluates the guidance.

Text Prompt: <prompt>"""

_prompts_0shot_control_image_gen_rule = """RULES:

Two images will be provided: The first being a processed image (e.g. Canny edges, openpose, grayscale etc.) and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_rule_PQ = """RULES:

The image is an AI-generated image.
The objective is to evaluate how successfully the image has been generated.

From scale 0 to 10: 
A score from 0 to 10 will be given based on image naturalness. 
(
    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. 
    10 indicates that the image looks natural.
)
A second score from 0 to 10 will rate the image artifacts. 
(
    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. 
    10 indicates the image has no artifacts.
)
Put the score in a list such that output score = [naturalness, artifacts]
"""

_prompts_0shot_t2v_rule_SC = """From scale 0 to 10: 
A score from 0 to 10 will be given based on the success in following the prompt. 
(0 indicates that the image frames does not follow the prompt at all. 10 indicates the image frames follows the prompt perfectly.)

Put the score in a list such that output score = [score].

Text Prompt: <prompt>
"""

_prompts_0shot_multi_subject_image_gen_rule = """RULES:

Two images will be provided: 
This first image is a concatenation of two sub-images, each sub-image contain one token subject.
The second image being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_subject_image_gen_rule = """RULES:

Two images will be provided: The first being a token subject image and the second being an AI-generated image using the first image as guidance.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_one_image_gen_rule = """RULES:

The image is an AI-generated image according to the text prompt.
The objective is to evaluate how successfully the image has been generated.
"""

_prompts_0shot_tie_rule_SC_move = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects (especially the moved object) and the scene background
in the final image match those in the original image.

**Scoring Criteria:**

* **0:** The moved object's identity and/or the scene background are completely inconsistent with the original image.
* **1-3:** The moved object or scene background is severely inconsistent.
Major identity changes, object replacement, or background reconstruction are observed.
* **4-6:** Some notable similarities exist, but clear inconsistencies remain.
The moved object partially changes identity or the background has noticeable alterations.
* **7-9:** The moved object identity and scene background are mostly consistent.
Only minor mismatches are present (small texture shifts, subtle feature deviations).
* **10:** The moved object identity and the scene background are perfectly consistent with the original image.
Only spatial position and/or scale has changed.

**Pay special attention to:**

* Whether the EXACT SAME object instance is preserved (not replaced with a similar-looking object).
* Object-specific attributes: shape, color, texture, material, fine-grained details.
* For humans or animals: facial/head features, hairstyle, body shape, skin tone, clothing (if not instructed to change).
* Whether any new object instance appears instead of the original one.
* Whether the background layout, lighting, perspective, and environmental objects remain unchanged.
* Whether unintended objects were modified.
* Whether the red box is completely removed in the final image.
* Identity consistency is the PRIMARY evaluation criterion.

**Important:**

* Only the spatial position and/or scale of the specified object is allowed to change.
* Any identity change of the object counts as a deduction.
* Any background alteration counts as a deduction.
* Every noticeable inconsistency deducts one point.
* Be strict. High scores (9-10) should only be given when identity preservation is nearly perfect.

Editing instruction: <instruction>
"""

_prompts_0shot_tie_rule_SC_rotate = """
You are a strict binary evaluator.

The user will provide ONE or TWO yes-or-no questions about an image (or a pair of images).

For EACH question:
- If the answer is YES, output 10
- If the answer is NO, output 0

The number of scores must EXACTLY match the number of questions provided.

Return your result in the following JSON format:

{
  "score": [x1, x2],
  "reasoning": "Brief explanation of the decisions."
}

Rules:
- If there is ONE question, output a list with ONE integer (e.g., [10]).
- If there are TWO questions, output a list with TWO integers (e.g., [10, 0]).
- Each integer must be either 0 or 10.
- The order of scores must strictly correspond to the order of the questions.
- Do NOT output anything outside the JSON.
- Do NOT add extra fields.
- Do NOT repeat the questions.
- Output valid JSON only.
- If a question cannot be answered with certainty, answer 0.
"""


_prompts_0shot_in_context_generation_rule_SC_Scene = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects and the scene background in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* with those in the reference images.
* **1-3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4-6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7-9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) and presence of key environmental details from the sence image.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""


_prompts_0shot_tie_rule_SC_camera_metric = """

The user will provide ONE pair of images.

Your task is to evaluate how similar their CAMERA VIEWPOINTS are.

Scoring rule:
- 10 = The viewpoints are visually identical (same yaw, pitch, distance, framing).
- 8–9 = Extremely similar viewpoint with only negligible differences.
- 6–7 = Clearly similar viewpoint but with noticeable rotation or distance shift.
- 4–5 = Moderately different viewpoint.
- 2–3 = Very different viewpoint.
- 0–1 = Completely different viewpoint (e.g., opposite direction or drastically different angle).

Important:
- Evaluate ONLY the camera viewpoint (yaw, pitch, distance).
- Ignore object appearance, lighting, texture, color differences.
- Ignore small object pose changes unless they indicate camera movement.
- Focus on perspective, horizon position, object relative projection, and depth cues.
- If uncertain, assign a lower score.

Return your result in the following JSON format:

{
  "score": x,
  "reasoning": "Brief explanation focusing only on viewpoint differences."
}

Rules:
- Output exactly ONE integer score between 0 and 10.
- Do NOT output a list.
- Do NOT output extra fields.
- Do NOT repeat the instructions.
- Output valid JSON only.
"""

SC_rotate = {
   "00": "Is {object_name} facing right side?",
   "01": "Is {object_name} facing right side?\nIs the front of {object_name} partially visible?",
   "02": "Is {object_name} facing the camera?",
   "03": "Is {object_name} facing left side?\nIs the front of {object_name} partially visible?",
   "04": "Is {object_name} facing left side?",
   "05": "Is {object_name} facing left side?\nIs the rear of {object_name} partially visible?",
   "06": "Is {object_name} facing away from the camera?",
   "07": "Is {object_name} facing right side?\nIs the rear of {object_name} partially visible?",
}


VIEW_PROMPT = {
    "00": "the right side view.",
    "01": "the front right side view.",
    "02": "the front view.",
    "03": "the front left side view.",
    "04": "the left side view.",
    "05": "the rear left side view.",
    "06": "the rear view.",
    "07": "the rear right side view.",
}