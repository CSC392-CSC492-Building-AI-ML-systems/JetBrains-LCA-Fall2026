```python
import argparse
import cv2
from ppstructure import PPStructureV2Table
from ppstructure.utility import DetectionPreprocessor, DetectionPostprocessor
from fastdeploy import RuntimeOption, ModelFormat, Backend, Device

def parse_args():
    parser = argparse.ArgumentParser(description="PPOCR Table Structure Inference")
    parser.add_argument('--det_model_path', type=str, required=True, help='Path to detection model directory')
    parser.add_argument('--rec_model_path', type=str, required=True, help='Path to recognition model directory')
    parser.add_argument('--table_model_path', type=str, required=True, help='Path to table recognition model directory')
    parser.add_argument('--rec_label_file', type=str, required=True, help='Path to recognition model label file')
    parser.add_argument('--table_char_dict_path', type=str, required=True, help='Path to table recognition dictionary')
    parser.add_argument('--rec_batch_num', type=int, default=6, help='Recognition model inference batch size')
    parser.add_argument('--image_path', type=str, required=True, help='Path to test image file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Inference device type')
    parser.add_argument('--device_id', type=int, default=0, help='Device ID')
    parser.add_argument('--backend', type=str, default='ort', choices=['ort', 'trt', 'paddle', 'openvino'], help='Inference backend type')
    return parser.parse_args()

def build_runtime_option(device, device_id, backend, model_name):
    option = RuntimeOption()
    if device == 'gpu':
        option.use_gpu(device_id)
    else:
        option.use_cpu()
    if backend == 'ort':
        option.use_ort_backend()
    elif backend == 'trt':
        option.use_trt_backend()
        # Set dynamic shape for TensorRT
        if model_name == 'det':
            option.set_trt_input_shape('x', [1, 3, 32, 32], [1, 3, 640, 640], [1, 3, 1280, 1280])
        elif model_name == 'rec':
            option.set_trt_input_shape('x', [1, 3, 32, 10], [1, 3, 32, 320], [1, 3, 32, 2304])
        elif model_name == 'table':
            option.set_trt_input_shape('x', [1, 3, 32, 32], [1, 3, 488, 488], [1, 3, 976, 976])
        # Save TRT cache
        option.set_trt_cache_file(f"{model_name}_trt_cache.trt")
    elif backend == 'paddle':
        option.use_paddle_backend()
    elif backend == 'openvino':
        option.use_openvino_backend()
    return option

def main():
    args = parse_args()

    # Build runtime options
    det_option = build_runtime_option(args.device, args.device_id, args.backend, 'det')
    rec_option = build_runtime_option(args.device, args.device_id, args.backend, 'rec')
    table_option = build_runtime_option(args.device, args.device_id, args.backend, 'table')

    # Load models
    from fastdeploy.vision.ocr import DBDetector, Classifier, Recognizer, TableRecognizer

    det_model = DBDetector(model_path=args.det_model_path, runtime_option=det_option, model_format=ModelFormat.ONNX)
    rec_model = Recognizer(model_path=args.rec_model_path, label_path=args.rec_label_file, runtime_option=rec_option, model_format=ModelFormat.ONNX)
    table_model = TableRecognizer(model_path=args.table_model_path, char_dict_path=args.table_char_dict_path, runtime_option=table_option, model_format=ModelFormat.ONNX)

    # Set preprocessor and postprocessor for detection model
    det_model.preprocessor = DetectionPreprocessor()
    det_model.postprocessor = DetectionPostprocessor()

    # Create PPStructureV2Table instance
    structure = PPStructureV2Table(det_model=det_model, rec_model=rec_model, table_model=table_model)
    structure.rec_batch_num = args.rec_batch_num

    # Read image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Failed to read image from {args.image_path}")

    # Predict and print results
    results = structure(image)
    print(results)

    # Visualize and save
    vis_img = structure.visualize(image, results)
    save_path = args.image_path.rsplit('.', 1)[0] + '_vis.jpg'
    cv2.imwrite(save_path, vis_img)
    print(f"Visualized image saved to {save_path}")

if __name__ == '__main__':
    main()
```