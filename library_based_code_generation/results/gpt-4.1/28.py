```python
import sys
import argparse
import numpy as np
import sirf.STIR as pet
import matplotlib.pyplot as plt

def osmaposl_reconstruction(image, objective_function, prior, image_filter, num_subsets, num_subiterations):
    for subiter in range(num_subiterations):
        for subset in range(num_subsets):
            # Compute gradient for current subset
            objective_function.set_subset_num(subset)
            gradient = objective_function.get_subset_gradient(image)
            # Add prior gradient if prior is set
            if prior is not None:
                prior_gradient = prior.get_gradient(image)
                gradient += prior_gradient
            # One Step Late update
            update = image.clone()
            update.fill(1.0)
            update -= gradient
            # Apply positivity constraint
            update.maximum(0.0)
            # Apply filter if provided
            if image_filter is not None:
                update = image_filter.apply(update)
            image.fill(update)
    return image

def main():
    parser = argparse.ArgumentParser(description='User-defined OSMAPOSL reconstruction with SIRF.')
    parser.add_argument('--raw_data_file', type=str, required=True, help='Raw data file name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to data files')
    parser.add_argument('--num_subsets', type=int, default=12, help='Number of subsets')
    parser.add_argument('--num_subiterations', type=int, default=10, help='Number of sub-iterations')
    parser.add_argument('--engine', type=str, default='STIR', help='Reconstruction engine')
    parser.add_argument('--no_plot', action='store_true', help='Disable plots')
    args = parser.parse_args()

    try:
        # Set up acquisition data
        pet.AcquisitionData.set_storage_scheme('memory')
        acq_data = pet.AcquisitionData(args.data_path + '/' + args.raw_data_file)

        # Set up acquisition model
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        template = acq_data.create_uniform_image(1.0)
        acq_model.set_up(acq_data, template)

        # Set up filter (optional, here using a simple Gaussian filter)
        image_filter = pet.TruncateToCylinderProcessor()
        # image_filter = None  # Uncomment to disable filtering

        # Set up prior (optional, here using quadratic prior)
        prior = pet.QuadraticPrior()
        prior.set_penalisation_factor(0.01)

        # Set up objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        obj_fun.set_num_subsets(args.num_subsets)
        if prior is not None:
            obj_fun.set_prior(prior)

        # Initial image estimate
        image = acq_data.create_uniform_image(1.0)

        # Run user-defined OSMAPOSL
        recon_image = osmaposl_reconstruction(
            image, obj_fun, prior, image_filter, args.num_subsets, args.num_subiterations
        )

        # Display result
        if not args.no_plot:
            arr = recon_image.as_array()
            plt.figure()
            plt.imshow(arr[arr.shape[0] // 2, :, :], cmap='gray')
            plt.title('Reconstructed Image (central slice)')
            plt.colorbar()
            plt.show()

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```