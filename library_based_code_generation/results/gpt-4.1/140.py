```python
import argparse
import cv2
import os
from paddlestructure import PPStructureV2Table
from paddlestructure.utils import draw_structure_result
from paddlestructure.ocr_det import DetectionModel
from paddlestructure.ocr_rec import RecognitionModel
from paddlestructure.ocr_table import TableRecognitionModel
from paddlestructure.runtime import RuntimeOption, BackendType, DeviceType

def parse_args():
    parser = argparse.ArgumentParser(description="PPOCR Table Structure Inference")
    parser.add_argument('--det_model_path', type=str, required=True, help='Path to detection model directory')
    parser.add_argument('--rec_model_path', type=str, required=True, help='Path to recognition model directory')
    parser.add_argument('--table_model_path', type=str, required=True, help='Path to table recognition model directory')
    parser.add_argument('--rec_label_file', type=str, required=True, help='Path to recognition model label file')
    parser.add_argument('--table_dict_path', type=str, required=True, help='Path to table recognition dictionary file')
    parser.add_argument('--rec_batch_num', type=int, default=6, help='Recognition model inference batch size')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Inference device type')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID')
    parser.add_argument('--backend', type=str, default='onnx_runtime', choices=['onnx_runtime', 'paddle', 'tensorrt'], help='Inference backend type')
    return parser.parse_args()

def build_runtime_option(device, device_id, backend, model_name):
    option = RuntimeOption()
    if device == 'gpu':
        option.use_gpu(device_id)
    else:
        option.use_cpu()
    if backend == 'onnx_runtime':
        option.use_onnx_runtime()
    elif backend == 'paddle':
        option.use_paddle_infer()
    elif backend == 'tensorrt':
        option.use_trt_backend()
        # Set dynamic shape for TRT
        if model_name == 'det':
            option.set_trt_input_shape('x', [1, 3, 640, 640], [1, 3, 1024, 1024], [1, 3, 1920, 1920])
        elif model_name == 'rec':
            option.set_trt_input_shape('x', [1, 3, 32, 10], [args.rec_batch_num, 3, 32, 320], [args.rec_batch_num, 3, 32, 2304])
        elif model_name == 'table':
            option.set_trt_input_shape('x', [1, 3, 488, 488], [1, 3, 488, 488], [1, 3, 488, 488])
        # Save TRT cache
        trt_cache = f"{model_name}_trt_cache_{device_id}.bin"
        option.set_trt_cache_file(trt_cache)
    return option

if __name__ == '__main__':
    args = parse_args()

    # Build runtime options
    det_option = build_runtime_option(args.device, args.device_id, args.backend, 'det')
    rec_option = build_runtime_option(args.device, args.device_id, args.backend, 'rec')
    table_option = build_runtime_option(args.device, args.device_id, args.backend, 'table')

    # Load models
    det_model = DetectionModel(args.det_model_path, runtime_option=det_option)
    rec_model = RecognitionModel(args.rec_model_path, label_file=args.rec_label_file, runtime_option=rec_option)
    table_model = TableRecognitionModel(args.table_model_path, dict_path=args.table_dict_path, runtime_option=table_option)

    # Set preprocessor and postprocessor parameters for detection model
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.box_thresh = 0.6
    det_model.postprocessor.unclip_ratio = 1.5

    # Create PPStructureV2Table instance
    structure = PPStructureV2Table(det_model, rec_model, table_model)
    structure.rec_batch_num = args.rec_batch_num

    # Read input image
    img = cv2.imread(args.image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Predict and print results
    results = structure(img)
    print(results)

    # Visualize and save results
    vis_img = draw_structure_result(img, results, font_path=None)
    save_path = os.path.splitext(args.image_path)[0] + "_structure_vis.jpg"
    cv2.imwrite(save_path, vis_img)
```