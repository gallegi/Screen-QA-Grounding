import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

ANN_PATH = "/teamspace/studios/this_studio/screen_qa-main/answers_and_bboxes/validation.json"
IMG_DIR = "/teamspace/studios/this_studio/screen_qa-main/RICO/combined"
SAVE_DIR = "/teamspace/studios/this_studio/outputs/qwen2.5_vl_pretrained"
os.makedirs(SAVE_DIR, exist_ok=True)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)
# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

PROMPT_TEMPLATE = """
    Answer the question based on the image.
    The outout format is JSON with following schema:
    {{
        "full_answer":{{
            "type": "string",
            "description": "The answer to the question."
        }},
        "bboxes": [
            {{
                "type": "list of list of 4 numbers",
                "description": "The bounding box (x1,y1,x2,y2) of the UI element where you base your answer on.",
            }}
        ]
    }}
    Example: 
    {{
        "full_answer": "The shiba inu is located in the top-left corner of the image.",
        "bboxes": [
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ]
    }}
    If the answer is not in the image, return {{"full_answer": "<no answer>", "bboxes": []}}
    Question: {question}
"""


with open(ANN_PATH, "r") as f:
    annotations = json.load(f)


ANNOTATOR_COLORS = [(147,112,219), (255,140,0), (32,178,170), (220,20,60), (0,139,139)]


# for idx in tqdm(range(len(annotations))):
for idx in np.random.randint(0, len(annotations), 3):
    ann = annotations[idx]
    quest = ann['question']

    img_path = os.path.join(IMG_DIR, f'{ann["image_id"]}.jpg')
    img = cv2.imread(img_path)
    draw = img.copy()

    print(f"Processing {idx}th image: {img_path}")

    for j, ans in enumerate(ann['ground_truth']):
        print(f"Annotator {j}: {ans['full_answer']}")
        draw = img.copy()
        for element in ans['ui_elements']:
            x1,y1,x2,y2 = element['bounds']
            color = ANNOTATOR_COLORS[j]
            draw = cv2.rectangle(draw, (x1,y1), (x2,y2), color, 3)


    # make prediction
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {"type": "text", "text": PROMPT_TEMPLATE.format(question=quest)},
            ],
        }
    ]
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    json_res = json.loads(output_text[0].replace("```json", "").replace("```", ""))
    print(f"Prediction: {json_res}")

    for box in json_res['bboxes']:
        draw = cv2.rectangle(draw, (box[0],box[1]), (box[2],box[3]), (255,0,0), 3) # prediction box: blue

    write_path = os.path.join(SAVE_DIR, f'{idx}.jpg')
    cv2.imwrite(write_path, draw)