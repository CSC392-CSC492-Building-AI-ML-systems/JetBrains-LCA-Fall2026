```python
import sys
import argparse
import numpy as np

try:
    import sirf.STIR as pet
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="PET Scatter Estimation using SIRF")
    parser.add_argument('--raw', required=True, help='Raw PET data file')
    parser.add_argument('--randoms', required=True, help='Randoms data file')
    parser.add_argument('--acf', required=True, help='Attenuation correction factors file')
    parser.add_argument('--norm_path', required=True, help='Path to normalization and attenuation files')
    parser.add_argument('--norm', required=True, help='Normalization file')
    parser.add_argument('--attn_image', required=True, help='Attenuation image file')
    parser.add_argument('--output_prefix', required=True, help='Output prefix for scatter estimates')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    args = parser.parse_args()

    try:
        # Read raw data
        raw_data = pet.AcquisitionData(args.raw)
        # Read randoms
        randoms = pet.AcquisitionData(args.randoms)
        # Read attenuation correction factors
        acf = pet.AcquisitionData(args.acf)
        # Set up normalization
        norm = pet.AcquisitionData(args.norm)
        # Set up attenuation image
        attn_image = pet.ImageData(args.attn_image)

        # Set up scatter estimator
        scatter_estimator = pet.ScatterEstimator()
        scatter_estimator.set_input(raw_data)
        scatter_estimator.set_randoms(randoms)
        scatter_estimator.set_attenuation_correction_factors(acf)
        scatter_estimator.set_normalisation(norm)
        scatter_estimator.set_attenuation_image(attn_image)
        scatter_estimator.set_path_to_attenuation_image(args.norm_path)
        scatter_estimator.set_path_to_normalisation_file(args.norm_path)

        # Estimate scatter
        scatter_estimate = scatter_estimator.estimate_scatter()
        scatter_filename = f"{args.output_prefix}_scatter.h5"
        scatter_estimate.write(scatter_filename)
        print(f"Scatter estimate written to {scatter_filename}")

        if not args.non_interactive:
            # Display scatter estimate
            scatter_array = scatter_estimate.as_array()
            plt.figure(figsize=(10, 4))
            plt.imshow(scatter_array[scatter_array.shape[0] // 2, :, :], cmap='hot')
            plt.title('Scatter Estimate (central slice)')
            plt.colorbar()
            plt.show()

            # Plot sinogram profile
            profile = scatter_array[scatter_array.shape[0] // 2, scatter_array.shape[1] // 2, :]
            plt.figure()
            plt.plot(profile)
            plt.title('Sinogram Profile (central view, central axial)')
            plt.xlabel('Bin')
            plt.ylabel('Scatter Estimate')
            plt.show()

    except Exception as e:
        print(f"Error during scatter estimation: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```