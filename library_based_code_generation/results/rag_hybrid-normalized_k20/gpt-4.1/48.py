```python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import sirf.STIR as pet
except ImportError:
    print("SIRF STIR module not found. Please install SIRF with STIR support.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Estimate randoms from listmode and compare with delayed coincidences.")
    parser.add_argument('--listmode-file', required=True, help='Path to the listmode file')
    parser.add_argument('--sinograms-file-prefix', required=True, help='Prefix for output sinograms files')
    parser.add_argument('--randoms-file', required=True, help='Output file for estimated randoms')
    parser.add_argument('--raw-data-template', required=True, help='Raw data template file')
    parser.add_argument('--data-path', required=True, help='Path to data files')
    parser.add_argument('--time-interval', nargs=2, type=float, metavar=('START', 'END'), default=None, help='Scanning time interval (start end)')
    parser.add_argument('--recon-engine', default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage-scheme', default='memory', choices=['memory', 'file'], help='Acquisition data storage scheme')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode (no plots)')
    args = parser.parse_args()

    try:
        # Set up ListmodeToSinograms converter
        lm2sino = pet.ListmodeToSinograms()
        lm2sino.set_input(args.listmode_file)
        lm2sino.set_output_file_prefix(args.sinograms_file_prefix)
        lm2sino.set_template(args.raw_data_template)
        if args.time_interval:
            lm2sino.set_time_interval(args.time_interval[0], args.time_interval[1])
        lm2sino.set_store_delayed(True)
        lm2sino.set_store_prompt(True)
        lm2sino.set_data_path(args.data_path)
        lm2sino.set_storage_scheme(args.storage_scheme)

        print("Processing listmode data to sinograms...")
        lm2sino.process()

        # Access sinograms
        print("Accessing sinograms...")
        prompts = lm2sino.get_output()
        delayeds = lm2sino.get_delayed_output()

        # Estimate randoms from delayeds via ML estimation
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
        print(f"  Prompts:  {prompts_arr.shape}")
        print(f"  Delayeds: {delayeds_arr.shape}")
        print(f"  Randoms:  {randoms_arr.shape}")
        print(f"Total number of delayed coincidences: {np.sum(delayeds_arr)}")
        print(f"Total number of estimated randoms:    {np.sum(randoms_arr)}")
        print(f"Max delayed: {np.max(delayeds_arr)}")
        print(f"Max randoms: {np.max(randoms_arr)}")

        # Display a single sinogram if not in non-interactive mode
        if not args.non_interactive:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(delayeds_arr[0], cmap='hot')
            plt.title('Delayeds (slice 0)')
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(randoms_arr[0], cmap='hot')
            plt.title('Estimated Randoms (slice 0)')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```