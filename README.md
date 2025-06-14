# ComfyUI Qwen2.5-VL Object Detection Node

This repository provides a custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for running object detection with the [Qwen 2.5 VL](https://github.com/QwenLM/Qwen2.5-VL) model. The node downloads the selected model on demand, runs a detection prompt and outputs bounding boxes that can be used with segmentation nodes such as [SAM2](https://github.com/kijai/ComfyUI-segment-anything-2).

## Nodes

### `DownloadAndLoadQwenModel`
Downloads a chosen Qwen 2.5-VL model into `models/Qwen` and returns the loaded model and processor. You can choose which device to load the model onto (e.g. `cuda:1` if you have multiple GPUs) and the precision for the checkpoint (INT4, INT8, BF16, FP16 or FP32).  FlashAttention is used for FP16/BF16 but FP32 falls back to PyTorch SDPA since FlashAttention does not support it.

### `QwenVLDetection`
Runs a detection prompt on an input image using the loaded model. The node outputs a JSON list of bounding boxes of the form `{"bbox_2d": [x1, y1, x2, y2], "label": "object"}` and a separate list of coordinates. Boxes are sorted by confidence and you can specify which ones to return using the **bbox_selection** parameter:

- `all` – return all boxes (default)
- Comma-separated indices such as `0`, `1,2` or `0,2` – return only the selected boxes, sorted by detection confidence
- `merge_boxes` – when enabled, merge the selected boxes into a single bounding box
- `score_threshold` – drop boxes with a confidence score below this value when available


The bounding boxes are converted to absolute pixel coordinates so they can be passed to SAM2 nodes.

### `BBoxesToSAM2`
Wraps a list of bounding boxes into the `BBOXES` batch format expected by
[`ComfyUI-segment-anything-2`](https://github.com/kijai/ComfyUI-segment-anything-2)
and compatible nodes such as
[`sam_2_ultra.py`](https://github.com/chflame163/ComfyUI_LayerStyle_Advance/blob/main/py/sam_2_ultra.py).

## Usage
1. Place this repository inside your `ComfyUI/custom_nodes` directory.
2. From the **Download and Load Qwen2.5-VL Model** node, select the model you want to use, choose the desired precision (INT4/INT8/BF16/FP16/FP32) and, if necessary, choose the device (such as `cuda:1`) where it should be loaded. The snapshot download will resume automatically if a previous attempt was interrupted.
3. Connect the output model to **Qwen2.5-VL Object Detection**, provide an image and the object you want to locate (e.g. `cat`). Optionally set **score_threshold** to filter out low-confidence boxes, use **bbox_selection** to choose specific ones (e.g. `0,2`) and enable **merge_boxes** if you want them merged. The node will automatically build the detection prompt and return the selected boxes in JSON.
4. Pass the bounding boxes through **Prepare BBoxes for SAM2** before feeding them into the SAM2 workflow.
