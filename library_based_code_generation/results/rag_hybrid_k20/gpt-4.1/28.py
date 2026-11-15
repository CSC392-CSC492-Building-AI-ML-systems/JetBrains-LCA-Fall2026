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
    if prior is not None:
        recon.set_prior(prior)
    if img_filter is not None:
        recon.set_image_data_processor(img_filter)
    recon.set_input(image)
    recon.process()
    return recon.get_output()

def main():
    parser = argparse.ArgumentParser(
        description='User-defined OSMAPOSL reconstruction with SIRF.'
    )
    parser.add_argument('--raw_data_file', type=str, required=True,
                        help='Raw data file (e.g., sinogram).')
    parser.add_argument('--data_path', type=str, default='.',
                        help='Path to data files.')
    parser.add_argument('--num_subsets', type=int, default=12,
                        help='Number of subsets.')
    parser.add_argument('--num_subiterations', type=int, default=2,
                        help='Number of sub-iterations.')
    parser.add_argument('--engine', type=str, default='STIR',
                        help='Reconstruction engine (default: STIR).')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting of reconstructed image.')
    args = parser.parse_args()

    try:
        ut.set_verbose(1)
        # Set up acquisition data
        raw_data_path = ut.full_filename(args.data_path, args.raw_data_file)
        acq_data = pet.AcquisitionData(raw_data_path)

        # Set up acquisition model
        if args.engine.upper() == 'STIR':
            acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
        elif args.engine.upper() == 'PARALLELPROJ':
            acq_model = pet.AcquisitionModelUsingParallelproj()
        else:
            print(f"Unknown engine: {args.engine}")
            sys.exit(1)
        acq_model.set_acquisition_data(acq_data)

        # Initial image estimate
        image = pet.ImageData(acq_data)
        image.fill(1.0)

        # Filter (optional, e.g., Gaussian)
        img_filter = pet.SeparableGaussianImageFilter()
        img_filter.set_fwhms([3.0, 3.0, 3.0])
        img_filter.set_maximum_number_of_sigmas(3.0)

        # Prior (optional, e.g., quadratic prior)
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
                mid = arr.shape[0] // 2
                plt.imshow(arr[mid, :, :], cmap='gray')
                plt.title('Reconstructed Image (central slice)')
                plt.colorbar()
                plt.show()
            except ImportError:
                print("matplotlib not installed, cannot plot image.")
        else:
            print("Reconstruction complete. Plotting disabled.")

    except Exception as e:
        print(f"Error during reconstruction: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```