```python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import sirf.STIR as pet
except ImportError:
    print("SIRF.STIR module not found. Please install SIRF with STIR support.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Estimate randoms from listmode and compare with delayed coincidences.")
    parser.add_argument('--listmode_file', required=True, help='Path to the listmode file')
    parser.add_argument('--sinograms_prefix', required=True, help='Prefix for output sinograms files')
    parser.add_argument('--randoms_file', required=True, help='Output file for estimated randoms')
    parser.add_argument('--raw_data_template', required=True, help='Raw data template file')
    parser.add_argument('--data_path', required=True, help='Path to data files')
    parser.add_argument('--time_interval', nargs=2, type=float, metavar=('START', 'END'), default=None, help='Scanning time interval (start end)')
    parser.add_argument('--recon_engine', default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage_scheme', default='file', choices=['file', 'memory'], help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    args = parser.parse_args()

    try:
        # Set up ListmodeToSinograms converter
        lm2sino = pet.ListmodeToSinograms()
        lm2sino.set_input(args.listmode_file)
        lm2sino.set_output_file_prefix(args.sinograms_prefix)
        lm2sino.set_template(args.raw_data_template)
        if args.time_interval:
            lm2sino.set_time_interval(args.time_interval[0], args.time_interval[1])
        lm2sino.set_store_delayed(True)
        lm2sino.set_data_path(args.data_path)

        print("Processing listmode data to sinograms...")
        lm2sino.process()

        # Access the sinograms and delayed coincidences
        print("Loading prompt and delayed sinograms...")
        prompts = pet.AcquisitionData(args.sinograms_prefix + "_prompts.hs")
        delayeds = pet.AcquisitionData(args.sinograms_prefix + "_delayeds.hs")

        # Estimate randoms from delayeds using ML estimation
        print("Estimating randoms from delayeds...")
        randoms = pet.estimate_randoms(delayeds)

        # Write estimated randoms to file
        print(f"Writing estimated randoms to {args.randoms_file}...")
        randoms.write(args.randoms_file)

        # Copy acquisition data into numpy arrays
        prompts_arr = prompts.as_array()
        delayeds_arr = delayeds.as_array()
        randoms_arr = randoms.as_array()

        # Print acquisition data dimensions and statistics
        print("Acquisition data dimensions:")
        print(f"  Prompts: {prompts_arr.shape}")
        print(f"  Delayeds: {delayeds_arr.shape}")
        print(f"  Estimated Randoms: {randoms_arr.shape}")
        print(f"Total number of delayed coincidences: {np.sum(delayeds_arr)}")
        print(f"Total number of estimated randoms: {np.sum(randoms_arr)}")
        print(f"Max delayed coincidences: {np.max(delayeds_arr)}")
        print(f"Max estimated randoms: {np.max(randoms_arr)}")

        # Display a single sinogram if not in non-interactive mode
        if not args.non_interactive:
            slice_idx = prompts_arr.shape[0] // 2
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(prompts_arr[slice_idx], cmap='hot')
            plt.title('Prompts')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(delayeds_arr[slice_idx], cmap='hot')
            plt.title('Delayeds')
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(randoms_arr[slice_idx], cmap='hot')
            plt.title('Estimated Randoms')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```