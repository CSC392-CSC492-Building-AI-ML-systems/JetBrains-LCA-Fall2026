```python
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import sirf.STIR as pet
except ImportError:
    print("SIRF library not found. Please install SIRF-PET.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PET Scatter Estimation using SIRF")
    parser.add_argument('--raw', required=True, help='Raw PET data file')
    parser.add_argument('--randoms', required=True, help='Randoms data file')
    parser.add_argument('--acf', required=True, help='Attenuation correction factors file')
    parser.add_argument('--norm_path', required=True, help='Path to normalization and attenuation files')
    parser.add_argument('--norm_file', required=True, help='Normalization file')
    parser.add_argument('--attn_image', required=True, help='Attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Output prefix for scatter estimates')
    parser.add_argument('--non_interactive', action='store_true', help='Non-interactive mode')
    args = parser.parse_args()

    try:
        # Load acquisition data
        print("Loading raw data...")
        acq_data = pet.AcquisitionData(args.raw)
        print("Loading randoms data...")
        randoms_data = pet.AcquisitionData(args.randoms)
        print("Loading attenuation correction factors...")
        acf_data = pet.AcquisitionData(args.acf)

        # Load normalization and attenuation image
        print("Loading normalization file...")
        norm_file_path = args.norm_path + '/' + args.norm_file
        normalization = pet.AcquisitionSensitivityModel(norm_file_path)
        print("Loading attenuation image...")
        attn_image = pet.ImageData(args.attn_image)

        # Set up scatter estimator
        print("Setting up scatter estimator...")
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_input(acq_data)
        scatter_estimator.set_randoms(randoms_data)
        scatter_estimator.set_attenuation_correction_factors(acf_data)
        scatter_estimator.set_normalisation(normalization)
        scatter_estimator.set_attenuation_image(attn_image)

        # Perform scatter estimation
        print("Estimating scatter...")
        scatter_estimate = scatter_estimator.estimate_scatter()
        scatter_filename = f"{args.output_prefix}_scatter.h5"
        scatter_estimate.write(scatter_filename)
        print(f"Scatter estimate saved to {scatter_filename}")

        if not args.non_interactive:
            # Display scatter estimate
            print("Displaying scatter estimate...")
            scatter_array = scatter_estimate.as_array()
            mid_slice = scatter_array.shape[1] // 2
            plt.figure()
            plt.imshow(scatter_array[0, mid_slice, :, :], cmap='hot')
            plt.title('Scatter Estimate (mid-slice)')
            plt.colorbar()
            plt.show()

            # Plot sinogram profile
            print("Plotting sinogram profile...")
            profile = scatter_array[0, mid_slice, :, scatter_array.shape[3] // 2]
            plt.figure()
            plt.plot(profile)
            plt.title('Sinogram Profile (mid-slice, central bin)')
            plt.xlabel('Bin')
            plt.ylabel('Scatter')
            plt.show()

    except Exception as e:
        print(f"Error during scatter estimation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```