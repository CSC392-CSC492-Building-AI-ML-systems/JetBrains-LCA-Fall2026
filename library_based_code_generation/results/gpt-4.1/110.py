```python
#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import traceback

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Steepest ascent for Poisson log-likelihood using SIRF.')
    parser.add_argument('--engine', type=str, required=True,
                        help='Reconstruction engine (e.g., STIR, Gadgetron)')
    parser.add_argument('--raw_data_file', type=str, required=True,
                        help='Raw PET acquisition data file')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data files')
    parser.add_argument('--num_steps', type=int, default=5,
                        help='Number of steepest ascent steps')
    parser.add_argument('--locally_optimal', action='store_true',
                        help='Use locally optimal steepest ascent')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbosity')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show plots')
    return parser.parse_args()

def truncate_image(image):
    """Zeroes image voxels outside a cylinder of the same diameter as the image xy-section size."""
    arr = image.as_array()
    shape = arr.shape
    nx, ny = shape[-2], shape[-1]
    x = np.arange(nx) - nx / 2 + 0.5
    y = np.arange(ny) - ny / 2 + 0.5
    X, Y = np.meshgrid(x, y, indexing='ij')
    radius = min(nx, ny) / 2
    mask = (X**2 + Y**2) <= radius**2
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
        # Import the specified engine
        if args.engine.lower() == 'stir':
            import sirf.STIR as pet
        elif args.engine.lower() == 'gadgetron':
            import sirf.Gadgetron as pet
        else:
            raise ValueError(f"Unknown engine: {args.engine}")

        # Set data path
        pet.AcquisitionData.set_storage_scheme('file')
        pet.AcquisitionData.set_path(args.data_path)

        # Read acquisition data
        acq_data = pet.AcquisitionData(args.raw_data_file)
        if args.verbose:
            print("Loaded acquisition data.")

        # Create acquisition model
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        if args.verbose:
            print("Created acquisition model.")

        # Create initial image estimate
        image = acq_data.create_uniform_image(1.0)
        if args.verbose:
            print("Created initial image estimate.")

        # Truncate image outside cylinder
        image = truncate_image(image)
        if args.verbose:
            print("Truncated image outside cylinder.")

        # Create Poisson log-likelihood objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        if args.verbose:
            print("Created Poisson log-likelihood objective function.")

        # Subset gradient setup
        num_subsets = 1  # For simplicity, can be parameterized
        obj_fun.set_num_subsets(num_subsets)

        # Steepest ascent steps
        current_image = image.clone()
        for step in range(args.num_steps):
            subset_num = step % num_subsets
            grad = obj_fun.gradient(current_image, subset_num)
            grad_arr = grad.as_array()
            if args.locally_optimal:
                # Compute optimal step size (simple line search)
                direction = grad_arr
                step_size = 1.0
                # Backtracking line search (very basic)
                for _ in range(10):
                    test_image = current_image.clone()
                    test_image.fill(current_image.as_array() + step_size * direction)
                    ll_new = obj_fun.value(test_image, subset_num)
                    ll_old = obj_fun.value(current_image, subset_num)
                    if ll_new > ll_old:
                        break
                    step_size *= 0.5
                if args.verbose:
                    print(f"Step {step+1}: Locally optimal step size = {step_size}")
                new_arr = current_image.as_array() + step_size * direction
            else:
                # Fixed step size
                step_size = 1e-2
                new_arr = current_image.as_array() + step_size * grad_arr
                if args.verbose:
                    print(f"Step {step+1}: Fixed step size = {step_size}")

            # Ensure positivity
            new_arr = np.maximum(new_arr, 0)
            current_image.fill(new_arr)
            current_image = truncate_image(current_image)

            ll_value = obj_fun.value(current_image, subset_num)
            if args.verbose:
                print(f"Step {step+1}: Log-likelihood = {ll_value}")

            if args.show_plots:
                plt.figure()
                plt.imshow(current_image.as_array()[current_image.as_array().shape[0]//2, :, :], cmap='gray')
                plt.title(f'Step {step+1}')
                plt.colorbar()
                plt.show()

    except Exception as e:
        print("An error occurred:")
        print(str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```