import os
import ast
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import folder_paths


def parse_json(json_output: str) -> str:
    """Extract the JSON payload from a model response."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:])
            json_output = json_output.split("```", 1)[0]
            break
    try:
        parsed = json.loads(json_output)
        # Some responses wrap the JSON in a {"content": "..."} object.
        if isinstance(parsed, dict) and "content" in parsed:
            inner = parsed["content"]
            if isinstance(inner, str):
                json_output = inner
    except Exception:
        pass
    return json_output


def parse_boxes(
    text: str,
    img_width: int,
    img_height: int,
    input_w: int,
    input_h: int,
    score_threshold: float = 0.0,
) -> List[List[int]]:
    text = parse_json(text)
    try:
        data = ast.literal_eval(text)
    except Exception:
        end_idx = text.rfind('"}') + len('"}')
        truncated = text[:end_idx] + "]"
        data = ast.literal_eval(truncated)
    if isinstance(data, dict):
        inner = data.get("content")
        if isinstance(inner, str):
            try:
                data = ast.literal_eval(inner)
            except Exception:
                data = []
        else:
            data = []
    items: List[Tuple[float, List[int]]] = []
    for item in data:
        box = item.get("bbox_2d") or item.get("bbox") or item
        score = float(item.get("score", 0))
        y1, x1, y2, x2 = box[1], box[0], box[3], box[2]
        abs_y1 = int(y1 / input_h * img_height)
        abs_x1 = int(x1 / input_w * img_width)
        abs_y2 = int(y2 / input_h * img_height)
        abs_x2 = int(x2 / input_w * img_width)
        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        items.append((score, [abs_x1, abs_y1, abs_x2, abs_y2]))
    items.sort(key=lambda x: x[0], reverse=True)
    filtered = [box for sc, box in items if sc >= score_threshold]
    if not filtered and items:
        # SAM2 expects at least one bbox. If all boxes are filtered out,
        # fall back to the highest‑scoring one so downstream nodes don't
        # error out.
        filtered = [items[0][1]]
    return filtered


@dataclass
class QwenModel:
    model: Any
    processor: Any
    device: str


class DownloadAndLoadQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "Qwen/Qwen2.5-VL-3B-Instruct",
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "Qwen/Qwen2.5-VL-32B-Instruct",
                    "Qwen/Qwen2.5-VL-72B-Instruct",
                ], ),
                "device": ([
                    "auto",
                    "cuda:0",
                    "cuda:1",
                    "cpu",
                ], ),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "Qwen2.5-VL"

    def load(self, model_name: str, device: str):
        model_dir = os.path.join(folder_paths.models_dir, "Qwen", model_name.replace("/", "_"))
        # Always attempt download with resume enabled so an interrupted download
        # can be continued when the node is executed again.
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if device == "auto":
            device_map = "auto"
        elif device == "cpu":
            device_map = {"": "cpu"}
        else:
            device_map = {"": device}

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            # If loading fails (likely due to an incomplete download), force a
            # re-download and try again.
            snapshot_download(
                repo_id=model_name,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                force_download=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            )
        processor = AutoProcessor.from_pretrained(model_dir)
        return (QwenModel(model=model, processor=processor, device=device),)


class QwenVLDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWEN_MODEL",),
                "image": ("IMAGE",),
                "target": ("STRING", {"default": "object"}),
                "bbox_selection": ("STRING", {"default": "all"}),
                "merge_boxes": ("BOOLEAN", {"default": False}),
                "score_threshold": ("FLOAT", {"default": 0.0}),
            },
        }

    RETURN_TYPES = ("JSON", "BBOX")
    RETURN_NAMES = ("text", "bboxes")
    FUNCTION = "detect"
    CATEGORY = "Qwen2.5-VL"

    def detect(
        self,
        qwen_model: QwenModel,
        image,
        target: str,
        bbox_selection: str = "all",
        merge_boxes: bool = False,
        score_threshold: float = 0.0,
    ):
        model = qwen_model.model
        processor = qwen_model.processor
        device = next(model.parameters()).device
        if qwen_model.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.set_device(int(qwen_model.device.split(":")[1]))
            except Exception:
                pass

        prompt = f"Locate the {target} and output bbox in JSON"

        if isinstance(image, torch.Tensor):
            image = (image.squeeze().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("Unsupported image type")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [{"type": "text", "text": prompt}, {"image": image}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        output_ids = model.generate(**inputs, max_new_tokens=1024)
        gen_ids = [output_ids[len(inp):] for inp, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        input_h = inputs['image_grid_thw'][0][1] * 14
        input_w = inputs['image_grid_thw'][0][2] * 14
        boxes = parse_boxes(
            output_text,
            image.width,
            image.height,
            input_w,
            input_h,
            score_threshold,
        )

        selection = bbox_selection.strip().lower()
        if selection != "all" and selection:
            idxs = []
            for part in selection.replace(",", " ").split():
                try:
                    idxs.append(int(part))
                except Exception:
                    continue
            boxes = [boxes[i] for i in idxs if 0 <= i < len(boxes)]

        if merge_boxes and boxes:
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            boxes = [[x1, y1, x2, y2]]

        return (output_text, boxes)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadQwenModel": DownloadAndLoadQwenModel,
    "QwenVLDetection": QwenVLDetection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadQwenModel": "Download and Load Qwen2.5-VL Model",
    "QwenVLDetection": "Qwen2.5-VL Object Detection",
}
