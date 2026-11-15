```python
import sys
import argparse
import numpy as np
import sirf.STIR as pet
import sirf.Utilities as ut

def osmaposl_reconstruction(
    image, objective_function, prior, image_filter, num_subsets, num_subiterations
):
    for subiter in range(num_subiterations):
        for subset in range(num_subsets):
            # Set subset for objective function
            objective_function.set_subset_num(subset)
            # Compute gradient
            gradient = objective_function.gradient(image)
            # Add prior gradient if prior is set
            if prior is not None:
                prior_grad = prior.gradient(image)
                gradient += prior_grad
            # OSL update: image = image - step_size * gradient
            # For simplicity, use step_size=1 (should be tuned in practice)
            image -= gradient
            # Apply image filter if provided
            if image_filter is not None:
                image_filter.apply(image)
    return image

def main():
    parser = argparse.ArgumentParser(
        description="User-defined OSMAPOSL reconstruction with SIRF"
    )
    parser.add_argument("--raw_data_file", type=str, required=True,
                        help="Raw data file (e.g., sinogram)")
    parser.add_argument("--data_path", type=str, default=".",
                        help="Path to data files")
    parser.add_argument("--num_subsets", type=int, default=12,
                        help="Number of subsets")
    parser.add_argument("--num_subiterations", type=int, default=24,
                        help="Number of sub-iterations")
    parser.add_argument("--engine", type=str, default="STIR",
                        help="Reconstruction engine (default: STIR)")
    parser.add_argument("--no_plot", action="store_true",
                        help="Disable plotting of reconstructed image")
    args = parser.parse_args()

    try:
        ut.set_verbose(1)
        # Set engine
        pet.AcquisitionData.set_storage_scheme('file')
        # Read acquisition data
        acq_data = pet.AcquisitionData(args.raw_data_file)
        # Create acquisition model
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        # Create initial image estimate
        image = acq_data.create_uniform_image(1.0)
        # Create prior (optional, e.g., QuadraticPrior)
        prior = pet.QuadraticPrior()
        prior.set_penalisation_factor(0.01)
        # Create objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        obj_fun.set_num_subsets(args.num_subsets)
        # Create image filter (optional, e.g., Gaussian)
        image_filter = pet.GaussianFilter()
        image_filter.set_fwhms([3.0, 3.0, 3.0])
        # Run OSMAPOSL reconstruction
        recon_image = osmaposl_reconstruction(
            image, obj_fun, prior, image_filter,
            args.num_subsets, args.num_subiterations
        )
        # Show result
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt
                arr = recon_image.as_array()
                plt.imshow(arr[arr.shape[0]//2, :, :], cmap='gray')
                plt.title('Reconstructed Image (central slice)')
                plt.show()
            except Exception as e:
                print("Plotting failed:", e)
        print("Reconstruction completed successfully.")
    except Exception as e:
        print("Error during reconstruction:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
```