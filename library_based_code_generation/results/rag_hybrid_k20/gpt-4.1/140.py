```python
import argparse
import cv2
from ppocr.runtime import (
    RuntimeOption,
    set_model_path,
    should_set_tensorrt,
    read_model,
    PPStructureV2Table,
    DBPostprocessor,
    DBPreprocessor,
    RecognizerPreprocessor,
    RecognizerPostprocessor,
    TableRecognizerPreprocessor,
    TableRecognizerPostprocessor
)

def parse_args():
    parser = argparse.ArgumentParser(description="PPOCR Structure Table Inference")
    parser.add_argument('--det_model_path', type=str, required=True, help='Path to detection model directory')
    parser.add_argument('--rec_model_path', type=str, required=True, help='Path to recognition model directory')
    parser.add_argument('--table_model_path', type=str, required=True, help='Path to table recognition model directory')
    parser.add_argument('--rec_label_file', type=str, required=True, help='Path to recognition model label file')
    parser.add_argument('--table_dict_path', type=str, required=True, help='Path to table recognition dictionary file')
    parser.add_argument('--rec_batch_num', type=int, default=6, help='Recognition model inference batch size')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Inference device type')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID')
    parser.add_argument('--backend', type=str, default='onnxruntime', choices=['onnxruntime', 'tensorrt', 'paddle'], help='Inference backend type')
    return parser.parse_args()

def build_runtime_option(model_path, device, device_id, backend, trt_cache_file=None):
    option = RuntimeOption()
    set_model_path(option, model_path)
    if device == 'gpu':
        option.use_gpu(device_id)
    else:
        option.use_cpu()
    if backend == 'onnxruntime':
        option.use_onnx_runtime()
    elif backend == 'tensorrt':
        option.use_trt_backend()
        if trt_cache_file:
            option.set_trt_cache_file(trt_cache_file)
            option.enable_trt_fp16()
            option.set_trt_input_shape('x', [1, 3, 32, 32], [1, 3, 640, 640], [1, 3, 1280, 1280])
    elif backend == 'paddle':
        option.use_paddle_infer()
    return option

def main():
    args = parse_args()

    # Build runtime options
    det_trt_cache = args.det_model_path + "/det_trt_cache.trt"
    rec_trt_cache = args.rec_model_path + "/rec_trt_cache.trt"
    table_trt_cache = args.table_model_path + "/table_trt_cache.trt"

    det_option = build_runtime_option(
        args.det_model_path, args.device, args.device_id, args.backend,
        trt_cache_file=det_trt_cache if args.backend == 'tensorrt' else None
    )
    rec_option = build_runtime_option(
        args.rec_model_path, args.device, args.device_id, args.backend,
        trt_cache_file=rec_trt_cache if args.backend == 'tensorrt' else None
    )
    table_option = build_runtime_option(
        args.table_model_path, args.device, args.device_id, args.backend,
        trt_cache_file=table_trt_cache if args.backend == 'tensorrt' else None
    )

    # Load models
    det_model = read_model(det_option)
    rec_model = read_model(rec_option)
    table_model = read_model(table_option)

    # Set preprocessor and postprocessor for detection model
    det_model.preprocessor = DBPreprocessor()
    det_model.postprocessor = DBPostprocessor()

    # Set preprocessor and postprocessor for recognition model
    rec_model.preprocessor = RecognizerPreprocessor(label_file=args.rec_label_file)
    rec_model.postprocessor = RecognizerPostprocessor(label_file=args.rec_label_file)

    # Set preprocessor and postprocessor for table recognition model
    table_model.preprocessor = TableRecognizerPreprocessor(dict_path=args.table_dict_path)
    table_model.postprocessor = TableRecognizerPostprocessor(dict_path=args.table_dict_path)

    # Create PPStructureV2Table instance
    structure = PPStructureV2Table(
        det_model=det_model,
        rec_model=rec_model,
        table_model=table_model
    )
    structure.rec_batch_num = args.rec_batch_num

    # Read input image
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Predict and print results
    results = structure(image)
    print(results)

    # Visualize and save result
    vis_img = structure.visualize(image, results)
    save_path = args.image_path + "_structure_vis.jpg"
    cv2.imwrite(save_path, vis_img)
    print(f"Visualized image saved to {save_path}")

if __name__ == "__main__":
    main()
```