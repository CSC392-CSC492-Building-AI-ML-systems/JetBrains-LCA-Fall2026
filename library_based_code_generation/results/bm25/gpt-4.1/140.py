```python
import argparse
import cv2
from ppstructure import PPStructureV2Table
from ppstructure.utility import draw_structure_result
from paddlestructure.runtime import RuntimeOption, Backend, Device

def parse_args():
    parser = argparse.ArgumentParser(description="PPOCR Structure Table Inference")
    parser.add_argument('--det_model_path', type=str, required=True, help='Path to detection model')
    parser.add_argument('--rec_model_path', type=str, required=True, help='Path to recognition model')
    parser.add_argument('--table_model_path', type=str, required=True, help='Path to table recognition model')
    parser.add_argument('--rec_label_file', type=str, required=True, help='Path to recognition model label file')
    parser.add_argument('--table_dict_path', type=str, required=True, help='Path to table recognition dictionary')
    parser.add_argument('--rec_batch_num', type=int, default=6, help='Recognition model inference batch size')
    parser.add_argument('--image_path', type=str, required=True, help='Test image file path')
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
        # Set dynamic shape for TensorRT
        if model_name == 'det':
            option.set_trt_input_shape('x', [1, 3, 640, 640], [1, 3, 960, 960], [1, 3, 1280, 1280])
            option.set_trt_cache_file('det_model.trt')
        elif model_name == 'rec':
            option.set_trt_input_shape('x', [1, 3, 32, 10], [args.rec_batch_num, 3, 32, 320], [args.rec_batch_num, 3, 32, 640])
            option.set_trt_cache_file('rec_model.trt')
        elif model_name == 'table':
            option.set_trt_input_shape('x', [1, 3, 488, 488], [1, 3, 488, 488], [1, 3, 488, 488])
            option.set_trt_cache_file('table_model.trt')
    return option

if __name__ == '__main__':
    args = parse_args()

    det_option = build_runtime_option(args.device, args.device_id, args.backend, 'det')
    rec_option = build_runtime_option(args.device, args.device_id, args.backend, 'rec')
    table_option = build_runtime_option(args.device, args.device_id, args.backend, 'table')

    # Load models
    from ppstructure.ocr_det import PPStructureDetModel
    from ppstructure.ocr_rec import PPStructureRecModel
    from ppstructure.table_rec import PPStructureTableModel

    det_model = PPStructureDetModel(args.det_model_path, runtime_option=det_option)
    rec_model = PPStructureRecModel(args.rec_model_path, args.rec_label_file, runtime_option=rec_option)
    table_model = PPStructureTableModel(args.table_model_path, args.table_dict_path, runtime_option=table_option)

    # Set preprocessor and postprocessor parameters for detection model
    det_model.preprocessor.max_side_len = 960
    det_model.postprocessor.box_thresh = 0.6
    det_model.postprocessor.unclip_ratio = 1.5

    # Create PPStructureV2Table instance
    structure = PPStructureV2Table(det_model=det_model, rec_model=rec_model, table_model=table_model)
    structure.rec_batch_num = args.rec_batch_num

    # Read image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {args.image_path}")

    # Predict
    results = structure(image)
    print(results)

    # Visualize and save
    vis_img = draw_structure_result(image, results, font_path=args.rec_label_file)
    save_path = args.image_path + "_vis.jpg"
    cv2.imwrite(save_path, vis_img)
```