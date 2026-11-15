```python
import sys
import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt

try:
    import sirf.STIR as pet
except ImportError:
    print("SIRF library not found. Please install SIRF and try again.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PET Scatter Estimation using SIRF")
    parser.add_argument('--raw_data', required=True, help='Path to raw PET data file')
    parser.add_argument('--randoms_data', required=True, help='Path to randoms data file')
    parser.add_argument('--acf_file', required=True, help='Path to attenuation correction factors file')
    parser.add_argument('--norm_attn_path', required=True, help='Path to normalization and attenuation files')
    parser.add_argument('--norm_file', required=True, help='Normalization file')
    parser.add_argument('--attn_image', required=True, help='Attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Output prefix for scatter estimates')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    args = parser.parse_args()

    try:
        # Read raw data
        acq_data = pet.AcquisitionData(args.raw_data)

        # Read randoms data
        randoms_data = pet.AcquisitionData(args.randoms_data)

        # Read attenuation correction factors
        acf_data = pet.AcquisitionData(args.acf_file)

        # Read normalization file
        norm_file_path = args.norm_attn_path + '/' + args.norm_file

        # Read attenuation image
        attn_image = pet.ImageData(args.attn_image)

        # Set up scatter estimator
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_input(acq_data)
        scatter_estimator.set_randoms(randoms_data)
        scatter_estimator.set_attenuation_correction_factors(acf_data)
        scatter_estimator.set_normalisation(norm_file_path)
        scatter_estimator.set_attenuation_image(attn_image)
        scatter_estimator.set_output_prefix(args.output_prefix)
        scatter_estimator.set_working_folder(args.output_prefix + '_scatter_working', overwrite=True)

        # Perform scatter estimation
        scatter_estimator.process()
        scatter_estimate = scatter_estimator.get_output()

        # Save scatter estimate
        scatter_estimate.write(args.output_prefix + '_scatter_estimate.h5')

        if not args.non_interactive:
            # Display scatter estimate
            scatter_estimate_array = scatter_estimate.as_array()
            mid_slice = scatter_estimate_array.shape[1] // 2
            plt.figure()
            plt.imshow(scatter_estimate_array[0, mid_slice, :, :], cmap='hot')
            plt.title('Scatter Estimate (sinogram mid-slice)')
            plt.colorbar()
            plt.show()

            # Plot sinogram profile
            profile = scatter_estimate_array[0, mid_slice, scatter_estimate_array.shape[2] // 2, :]
            plt.figure()
            plt.plot(profile)
            plt.title('Sinogram Profile (mid-slice)')
            plt.xlabel('Bin')
            plt.ylabel('Scatter Estimate')
            plt.show()

    except Exception as e:
        print("Error during scatter estimation:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
```