```python
import sys
import argparse
import os
import traceback

try:
    import sirf.STIR as pet
    import sirf.Utilities as ut
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Error importing SIRF or required libraries:", e)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PET scatter estimation using SIRF.")
    parser.add_argument('--raw', required=True, help='Raw PET data file')
    parser.add_argument('--randoms', required=True, help='Randoms data file')
    parser.add_argument('--acf', required=True, help='Attenuation correction factors file')
    parser.add_argument('--norm_path', required=True, help='Path to normalization and attenuation files')
    parser.add_argument('--norm_file', required=True, help='Normalization file')
    parser.add_argument('--attn_img', required=True, help='Attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Output prefix for scatter estimates')
    parser.add_argument('--non_interactive', action='store_true', help='Non-interactive mode')
    args = parser.parse_args()

    try:
        # Set up acquisition data
        print("Reading raw data...")
        acq_data = pet.AcquisitionData(args.raw)
        print("Reading randoms data...")
        randoms_data = pet.AcquisitionData(args.randoms)
        print("Reading attenuation correction factors...")
        acf_data = pet.AcquisitionData(args.acf)

        # Set up normalization
        norm_file_path = os.path.join(args.norm_path, args.norm_file)
        print("Setting up normalization from:", norm_file_path)
        norm = pet.AcquisitionSensitivityModel(norm_file_path)

        # Set up attenuation image
        print("Reading attenuation image...")
        attn_image = pet.ImageData(args.attn_img)

        # Set up scatter estimator
        print("Setting up scatter estimator...")
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_acquisition_data(acq_data)
        scatter_estimator.set_randoms(randoms_data)
        scatter_estimator.set_attenuation_correction_factors(acf_data)
        scatter_estimator.set_normalisation(norm)
        scatter_estimator.set_attenuation_image(attn_image)

        # Output file
        output_file = args.output_prefix + "_scatter_estimate.h5"

        # Estimate scatter
        print("Estimating scatter...")
        scatter_estimate = scatter_estimator.estimate()
        scatter_estimate.write(output_file)
        print(f"Scatter estimate written to {output_file}")

        if not args.non_interactive:
            # Display scatter estimate
            print("Displaying scatter estimate...")
            scatter_estimate_array = scatter_estimate.as_array()
            plt.figure()
            plt.imshow(scatter_estimate_array[0, :, :], cmap='hot')
            plt.title('Scatter Estimate (first sinogram)')
            plt.colorbar()
            plt.show()

            # Plot sinogram profile
            print("Plotting sinogram profile...")
            profile = scatter_estimate_array[0, scatter_estimate_array.shape[1] // 2, :]
            plt.figure()
            plt.plot(profile)
            plt.title('Sinogram Profile (middle row of first sinogram)')
            plt.xlabel('Bin')
            plt.ylabel('Scatter Estimate')
            plt.show()

    except Exception as e:
        print("An error occurred during scatter estimation:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
```