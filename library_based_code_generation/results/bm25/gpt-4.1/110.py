```python
import sys
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt

def truncate_image(image):
    from sirf.STIR import TruncateToCylinderProcessor
    processor = TruncateToCylinderProcessor()
    processor.set_strictly_less_than_radius(False)
    processor.apply(image)
    return image

def main():
    try:
        parser = argparse.ArgumentParser(description='Steepest ascent for Poisson log-likelihood using subset gradients.')
        parser.add_argument('--engine', type=str, required=True, help='Reconstruction engine (e.g., STIR)')
        parser.add_argument('--raw_data_file', type=str, required=True, help='Raw data file')
        parser.add_argument('--data_path', type=str, default='.', help='Path to data files')
        parser.add_argument('--num_steps', type=int, default=5, help='Number of steepest ascent steps')
        parser.add_argument('--locally_optimal', action='store_true', help='Use locally optimal steepest ascent')
        parser.add_argument('--verbose', action='store_true', help='Verbosity')
        parser.add_argument('--show_plots', action='store_true', help='Show plots')
        args = parser.parse_args()

        # Import engine module
        if args.engine.upper() == 'STIR':
            import sirf.STIR as pet
        else:
            raise ValueError(f"Unknown engine: {args.engine}")

        # Read acquisition data
        acq_data = pet.AcquisitionData(args.raw_data_file)

        # Create initial image estimate
        image = pet.ImageData()
        image = acq_data.create_uniform_image(1.0)

        # Truncate image to cylinder
        image = truncate_image(image)

        # Create acquisition model
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        acq_model.set_up(acq_data, image)

        # Create Poisson log-likelihood objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)

        # Number of subsets
        num_subsets = 1

        # Steepest ascent steps
        for step in range(args.num_steps):
            if args.verbose:
                print(f"Step {step+1}/{args.num_steps}")

            grad = obj_fun.get_subset_gradient(image, step % num_subsets, num_subsets)
            grad_arr = grad.as_array()
            if args.locally_optimal:
                # Compute optimal step size (simple line search)
                direction = grad_arr
                step_size = 1.0
                for _ in range(5):
                    test_image = image.clone()
                    test_image.fill(image.as_array() + step_size * direction)
                    val1 = obj_fun.value(image)
                    val2 = obj_fun.value(test_image)
                    if val2 > val1:
                        step_size *= 1.5
                    else:
                        step_size *= 0.5
                if args.verbose:
                    print(f"  Locally optimal step size: {step_size}")
            else:
                step_size = 1e-2

            image_arr = image.as_array()
            image_arr += step_size * grad_arr
            image.fill(image_arr)
            image = truncate_image(image)

            if args.show_plots:
                plt.figure()
                plt.imshow(image.as_array()[image.as_array().shape[0]//2,:,:], cmap='gray')
                plt.title(f'Slice at step {step+1}')
                plt.colorbar()
                plt.show(block=False)
                plt.pause(0.5)
                plt.close()

        if args.show_plots:
            plt.figure()
            plt.imshow(image.as_array()[image.as_array().shape[0]//2,:,:], cmap='gray')
            plt.title('Final image')
            plt.colorbar()
            plt.show()

    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
```