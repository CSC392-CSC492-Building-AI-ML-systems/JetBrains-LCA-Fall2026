```python
import sys
import argparse
import numpy as np
import traceback

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Steepest ascent for Poisson log-likelihood using subset gradients.')
    parser.add_argument('--engine', type=str, required=True, help='Reconstruction engine module (e.g., sirf.STIR)')
    parser.add_argument('--raw_data_file', type=str, required=True, help='Raw PET acquisition data file')
    parser.add_argument('--data_path', type=str, default='.', help='Path to data files')
    parser.add_argument('--num_steps', type=int, default=5, help='Number of steepest ascent steps')
    parser.add_argument('--locally_optimal', action='store_true', help='Use locally optimal steepest ascent')
    parser.add_argument('--verbose', action='store_true', help='Verbosity')
    parser.add_argument('--show_plots', action='store_true', help='Show plots')
    return parser.parse_args()

def truncate_image(image, mask):
    arr = image.as_array()
    arr[~mask] = 0
    image.fill(arr)
    return image

def create_cylinder_mask(image):
    arr = image.as_array()
    shape = arr.shape
    y, x = np.ogrid[:shape[1], :shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    radius = min(shape[1], shape[2]) // 2
    mask2d = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    mask = np.zeros(shape, dtype=bool)
    mask[:, :, :] = mask2d
    return mask

def main():
    args = parse_args()
    try:
        # Dynamically import the specified engine
        import importlib
        engine_mod = importlib.import_module(args.engine)
        # Import SIRF objects
        AcquisitionData = getattr(engine_mod, 'AcquisitionData')
        ImageData = getattr(engine_mod, 'ImageData')
        AcquisitionModel = getattr(engine_mod, 'AcquisitionModelUsingParallelproj', None)
        if AcquisitionModel is None:
            AcquisitionModel = getattr(engine_mod, 'AcquisitionModelUsingRayTracingMatrix')
        PoissonLogLikelihood = getattr(engine_mod, 'PoissonLogLikelihoodWithLinearModelForMeanAndProjData')
        ImageDataProcessor = getattr(engine_mod, 'ImageDataProcessor')

        # Read acquisition data
        acq_data = AcquisitionData(args.raw_data_file)
        if args.verbose:
            print('Loaded acquisition data:', acq_data)

        # Create acquisition model
        image_template = acq_data.create_uniform_image(1.0)
        acq_model = AcquisitionModel()
        acq_model.set_up(acq_data, image_template)

        # Create a mask for the cylinder
        mask = create_cylinder_mask(image_template)

        # Create a filter to zero outside the cylinder
        class CylinderTruncator(ImageDataProcessor):
            def __init__(self, mask):
                super().__init__()
                self.mask = mask
            def apply(self, image):
                truncate_image(image, self.mask)

        truncator = CylinderTruncator(mask)

        # Initial image estimate
        image = acq_data.create_uniform_image(1.0)
        truncator.apply(image)

        # Create Poisson log-likelihood objective function
        obj_fun = PoissonLogLikelihood()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        obj_fun.set_image_data_processor(truncator)

        # Steepest ascent steps
        for step in range(args.num_steps):
            if args.verbose:
                print(f'Step {step+1}/{args.num_steps}')
            obj_fun.set_up(image)
            grad = obj_fun.get_gradient(image)
            if args.locally_optimal:
                # Compute optimal step size along gradient direction
                def f(lam):
                    test_img = image.clone()
                    arr = test_img.as_array()
                    arr += lam * grad.as_array()
                    test_img.fill(arr)
                    return -obj_fun.value(test_img)
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(f, bounds=(0, 1), method='bounded')
                step_size = res.x
                if args.verbose:
                    print('Locally optimal step size:', step_size)
            else:
                # Use a fixed step size
                step_size = 1e-2
            arr = image.as_array()
            arr += step_size * grad.as_array()
            arr[arr < 0] = 0  # enforce non-negativity
            image.fill(arr)
            truncator.apply(image)
            if args.show_plots:
                plt.imshow(image.as_array()[image.as_array().shape[0]//2], cmap='gray')
                plt.title(f'Step {step+1}')
                plt.show()
        if args.verbose:
            print('Final image min/max:', image.as_array().min(), image.as_array().max())
    except Exception as e:
        print('An error occurred:', str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```