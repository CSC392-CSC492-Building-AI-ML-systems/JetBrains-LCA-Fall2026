```python
import sys
import argparse
import sirf.STIR as pet
import sirf.Utilities as ut
import numpy as np

def osmaposl_reconstruction(
    image, objective_function, prior, img_filter,
    num_subsets, num_subiterations
):
    recon = pet.OSMAPOSLReconstructor()
    recon.set_objective_function(objective_function)
    recon.set_num_subsets(num_subsets)
    recon.set_num_subiterations(num_subiterations)
    recon.set_input(image)
    if prior is not None:
        recon.set_prior(prior)
    if img_filter is not None:
        recon.set_image_data_processor(img_filter)
    recon.set_up()
    recon.process()
    return recon.get_output()

def main():
    parser = argparse.ArgumentParser(
        description='User-defined OSMAPOSL reconstruction with SIRF.'
    )
    parser.add_argument('--raw_data_file', type=str, required=True,
                        help='Raw data file name')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path to data files')
    parser.add_argument('--num_subsets', type=int, default=12,
                        help='Number of subsets')
    parser.add_argument('--num_subiterations', type=int, default=2,
                        help='Number of sub-iterations')
    parser.add_argument('--engine', type=str, default='STIR',
                        help='Reconstruction engine (default: STIR)')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting of reconstructed image')
    args = parser.parse_args()

    try:
        ut.set_verbose(1)
        # Load acquisition data
        acq_data = pet.AcquisitionData(args.raw_data_file)
        # Set up acquisition model
        acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        # Initial image estimate
        image = acq_data.create_uniform_image(1.0)
        # Optional: create a filter (e.g., Gaussian)
        img_filter = pet.TruncateToCylinderProcessor()
        # Optional: create a prior (e.g., quadratic prior)
        prior = pet.QuadraticPrior()
        prior.set_penalisation_factor(0.01)
        # Objective function
        obj_fun = pet.PoissonLogLikelihoodWithLinearModelForMeanAndProjData()
        obj_fun.set_acquisition_model(acq_model)
        obj_fun.set_acquisition_data(acq_data)
        # Run reconstruction
        recon_image = osmaposl_reconstruction(
            image, obj_fun, prior, img_filter,
            args.num_subsets, args.num_subiterations
        )
        # Display result
        if not args.no_plot:
            try:
                import matplotlib.pyplot as plt
                arr = recon_image.as_array()
                plt.imshow(arr[arr.shape[0]//2, :, :], cmap='gray')
                plt.title('Reconstructed image (central slice)')
                plt.colorbar()
                plt.show()
            except Exception as e:
                print(f"Plotting failed: {e}")
        print("Reconstruction completed successfully.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```