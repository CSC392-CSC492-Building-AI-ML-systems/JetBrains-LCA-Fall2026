```python
import sys
import argparse
import numpy as np
import traceback

import sirf
from sirf.Utilities import show_2D_array

def truncate_image(image, mask):
    """Zeroes the image outside the mask."""
    arr = image.as_array()
    arr[~mask] = 0
    image.fill(arr)
    return image

def create_cylinder_mask(image):
    """Creates a cylindrical mask for the image."""
    arr = image.as_array()
    shape = arr.shape
    if len(shape) == 3:
        z, y, x = np.indices(shape)
        cy, cx = shape[1] // 2, shape[2] // 2
        radius = min(shape[1], shape[2]) // 2
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
        mask = np.broadcast_to(mask, shape)
    elif len(shape) == 2:
        y, x = np.indices(shape)
        cy, cx = shape[0] // 2, shape[1] // 2
        radius = min(shape[0], shape[1]) // 2
        mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    else:
        raise ValueError("Unsupported image shape for mask creation.")
    return mask

def main():
    parser = argparse.ArgumentParser(description="Steepest ascent for Poisson log-likelihood using subset gradients.")
    parser.add_argument('--engine', type=str, required=True, help='Reconstruction engine (e.g., STIR, Parallelproj)')
    parser.add_argument('--raw_data_file', type=str, required=True, help='Raw data file')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data files')
    parser.add_argument('--num_steps', type=int, default=5, help='Number of steepest ascent steps')
    parser.add_argument('--locally_optimal', action='store_true', help='Use locally optimal steepest ascent')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    args = parser.parse_args()

    try:
        # Import engine module
        if args.engine.lower() == 'stir':
            import sirf.STIR as pet
        elif args.engine.lower() == 'parallelproj':
            import sirf.Parallelproj as pet
        else:
            raise ValueError(f"Unknown engine: {args.engine}")

        # Read acquisition data
        acq_data = pet.AcquisitionData(args.raw_data_file)
        if args.verbose:
            print("Acquisition data loaded.")

        # Create acquisition model
        if args.engine.lower() == 'stir':
            acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        elif args.engine.lower() == 'parallelproj':
            acq_model = pet.AcquisitionModelUsingParallelproj()
        else:
            raise ValueError(f"Unknown engine: {args.engine}")

        acq_model.set_acquisition_data(acq_data)
        if args.verbose:
            print("Acquisition model set.")

        # Create initial image estimate
        image = acq_data.create_uniform_image(1.0)
        if args.verbose:
            print("Initial image created.")

        # Create cylindrical mask and filter
        mask = create_cylinder_mask(image)
        image = truncate_image(image, mask)
        if args.verbose:
            print("Image truncated to cylinder.")

        # Create Poisson log-likelihood objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        if args.verbose:
            print("Objective function created.")

        # Optionally show initial image
        if args.show_plots:
            arr = image.as_array()
            if arr.ndim == 3:
                show_2D_array(arr[arr.shape[0] // 2, :, :], "Initial image (central slice)")
            else:
                show_2D_array(arr, "Initial image")

        # Steepest ascent steps
        current_image = image.clone()
        n_subsets = obj_fun.get_num_subsets() if hasattr(obj_fun, 'get_num_subsets') else 1
        for step in range(args.num_steps):
            if args.verbose:
                print(f"Step {step+1}/{args.num_steps}")
            for subset in range(n_subsets):
                grad = obj_fun.get_subset_gradient(current_image, subset)
                if args.locally_optimal:
                    # Compute optimal step size (simple line search)
                    alpha = 1.0
                    for _ in range(5):
                        test_image = current_image.clone()
                        arr = test_image.as_array() + alpha * grad.as_array()
                        arr[arr < 0] = 0
                        test_image.fill(arr)
                        ll = obj_fun.value(test_image)
                        ll0 = obj_fun.value(current_image)
                        if ll > ll0:
                            break
                        alpha *= 0.5
                    if args.verbose:
                        print(f"  Subset {subset}: locally optimal alpha={alpha}")
                else:
                    alpha = 1e-2
                arr = current_image.as_array() + alpha * grad.as_array()
                arr[arr < 0] = 0
                current_image.fill(arr)
                current_image = truncate_image(current_image, mask)
            if args.show_plots:
                arr = current_image.as_array()
                if arr.ndim == 3:
                    show_2D_array(arr[arr.shape[0] // 2, :, :], f"Image after step {step+1}")
                else:
                    show_2D_array(arr, f"Image after step {step+1}")

        if args.verbose:
            print("Steepest ascent completed.")

    except Exception as e:
        print("An error occurred:")
        print(str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```