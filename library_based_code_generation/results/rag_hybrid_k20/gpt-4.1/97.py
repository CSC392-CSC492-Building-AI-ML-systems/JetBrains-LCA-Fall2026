```python
import sys
import argparse
import os
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
    parser.add_argument('--norm_path', required=True, help='Path to normalization and attenuation files')
    parser.add_argument('--norm_file', required=True, help='Normalization file name')
    parser.add_argument('--attn_image', required=True, help='Attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Prefix for output scatter estimate files')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    args = parser.parse_args()

    try:
        # Load raw data
        acq_data = pet.AcquisitionData(args.raw_data)

        # Load randoms data
        randoms_data = pet.AcquisitionData(args.randoms_data)

        # Load attenuation correction factors
        acf_data = pet.AcquisitionData(args.acf_file)

        # Load normalization file
        norm_file_path = os.path.join(args.norm_path, args.norm_file)
        if not os.path.isfile(norm_file_path):
            raise FileNotFoundError(f"Normalization file not found: {norm_file_path}")

        # Load attenuation image
        attn_image = pet.ImageData(args.attn_image)

        # Set up scatter estimator
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_acquisition_data(acq_data)
        scatter_estimator.set_randoms(randoms_data)
        scatter_estimator.set_attenuation_correction_factors(acf_data)
        scatter_estimator.set_normalisation(norm_file_path)
        scatter_estimator.set_attenuation_image(attn_image)

        # Estimate scatter
        print("Estimating scatter...")
        scatter_estimate = scatter_estimator.estimate()

        # Save scatter estimate
        scatter_output_file = f"{args.output_prefix}_scatter.h5"
        scatter_estimate.write(scatter_output_file)
        print(f"Scatter estimate saved to {scatter_output_file}")

        if not args.non_interactive:
            # Display scatter estimate
            scatter_array = scatter_estimate.as_array()
            plt.figure(figsize=(10, 4))
            plt.imshow(scatter_array[0, :, :], cmap='hot')
            plt.title('Scatter Estimate (first sinogram segment)')
            plt.colorbar()
            plt.show()

            # Plot sinogram profile
            profile = scatter_array[0, scatter_array.shape[1] // 2, :]
            plt.figure()
            plt.plot(profile)
            plt.title('Sinogram Profile (middle axial bin)')
            plt.xlabel('Bin')
            plt.ylabel('Scatter Estimate')
            plt.show()

    except Exception as e:
        print(f"Error during scatter estimation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```