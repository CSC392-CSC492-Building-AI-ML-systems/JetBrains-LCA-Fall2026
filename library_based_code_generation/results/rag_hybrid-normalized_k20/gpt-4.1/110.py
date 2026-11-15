```python
import sys
import argparse
import numpy as np
import traceback

import sirf
from sirf.Utilities import show_2D_array

def parse_args():
    parser = argparse.ArgumentParser(description='Steepest ascent for Poisson log-likelihood using subset gradients.')
    parser.add_argument('--engine', type=str, required=True, help='Reconstruction engine (e.g., STIR, PET, etc.)')
    parser.add_argument('--raw_data_file', type=str, required=True, help='Raw data file name')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data files')
    parser.add_argument('--num_steps', type=int, default=5, help='Number of steepest ascent steps')
    parser.add_argument('--locally_optimal', action='store_true', help='Use locally optimal steepest ascent')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    return parser.parse_args()

def truncate_image(image):
    # Zero image outside a cylinder of the same diameter as the image xy-section size
    arr = image.as_array()
    shape = arr.shape
    nx, ny = shape[-2], shape[-1]
    x = np.arange(nx) - nx / 2 + 0.5
    y = np.arange(ny) - ny / 2 + 0.5
    xx, yy = np.meshgrid(x, y, indexing='ij')
    radius = min(nx, ny) / 2
    mask = (xx**2 + yy**2) <= radius**2
    if arr.ndim == 3:
        for z in range(arr.shape[0]):
            arr[z, :, :] *= mask
    elif arr.ndim == 2:
        arr *= mask
    image.fill(arr)
    return image

def main():
    args = parse_args()
    try:
        # Import engine module
        engine_module = getattr(sirf, args.engine)
        # Set up acquisition data
        acq_data = engine_module.AcquisitionData(args.data_path + '/' + args.raw_data_file)
        if args.verbose:
            print('Acquisition data loaded:', acq_data)
        # Set up acquisition model
        acq_model = engine_module.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_acquisition_data(acq_data)
        # Create initial image estimate
        image = acq_data.create_uniform_image(1.0)
        # Truncate image outside cylinder
        image = truncate_image(image)
        # Create Poisson log-likelihood objective function
        obj_fun = engine_module.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        # Optionally set image data processor (filter)
        # (already truncated above)
        # Preprocess acquisition data
        obj_fun.preprocess_acquisition_data(image)
        # Steepest ascent steps
        current_image = image.clone()
        for step in range(args.num_steps):
            if args.verbose:
                print(f'Step {step+1}/{args.num_steps}')
            # Compute gradient (subset gradients not implemented in this example)
            gradient = obj_fun.gradient(current_image)
            # Locally optimal step size
            if args.locally_optimal:
                # Line search for optimal step size (simple backtracking)
                step_size = 1.0
                best_value = obj_fun(current_image)
                for _ in range(10):
                    test_image = current_image.clone()
                    test_image.fill(current_image.as_array() + step_size * gradient.as_array())
                    value = obj_fun(test_image)
                    if value > best_value:
                        best_value = value
                        break
                    step_size *= 0.5
                if args.verbose:
                    print(f'  Locally optimal step size: {step_size}')
            else:
                step_size = 1e-2
            # Update image
            current_image.fill(current_image.as_array() + step_size * gradient.as_array())
            # Truncate image outside cylinder
            current_image = truncate_image(current_image)
            if args.verbose:
                print(f'  Objective function value: {obj_fun(current_image)}')
            if args.show_plots:
                arr = current_image.as_array()
                show_2D_array(arr[arr.shape[0]//2, :, :], title=f'Step {step+1}')
        if args.verbose:
            print('Steepest ascent completed.')
    except Exception as e:
        print('An error occurred:')
        print(str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```