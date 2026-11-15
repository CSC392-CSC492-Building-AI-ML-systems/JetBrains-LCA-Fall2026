```python
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import sirf.STIR as pet
except ImportError:
    print("SIRF STIR module not found. Please install SIRF.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Estimate randoms from listmode and compare with delayed coincidences.")
    parser.add_argument('--listmode', required=True, help='Path to listmode file')
    parser.add_argument('--sinograms_prefix', required=True, help='Prefix for output sinograms files')
    parser.add_argument('--randoms_file', required=True, help='Output file for estimated randoms')
    parser.add_argument('--raw_data_template', required=True, help='Raw data template file')
    parser.add_argument('--delayed_sinogram', required=True, help='Sinogram file for delayed coincidences')
    parser.add_argument('--time_interval', nargs=2, type=float, metavar=('START', 'END'), default=None, help='Scanning time interval (start end)')
    parser.add_argument('--engine', default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage_scheme', default='memory', help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    args = parser.parse_args()

    try:
        # Set up the listmode-to-sinograms converter
        lm2sino = pet.ListmodeToSinograms()
        lm2sino.set_input(args.listmode)
        lm2sino.set_output_file_prefix(args.sinograms_prefix)
        lm2sino.set_template(args.raw_data_template)
        if args.time_interval:
            lm2sino.set_time_interval(args.time_interval[0], args.time_interval[1])
        lm2sino.set_store_delayed_events(True)
        lm2sino.set_store_prompt_events(True)
        lm2sino.set_output_file_format('interfile')
        lm2sino.set_storage_scheme(args.storage_scheme)

        # Process the data
        print("Processing listmode data to sinograms...")
        lm2sino.process()

        # Access the sinograms
        prompt_sino_file = args.sinograms_prefix + "_prompts.hs"
        delayed_sino_file = args.sinograms_prefix + "_delayeds.hs"
        prompt_sino = pet.AcquisitionData(prompt_sino_file)
        delayed_sino = pet.AcquisitionData(delayed_sino_file)

        # Estimate randoms from delayeds via ML estimation
        print("Estimating randoms from delayed coincidences...")
        estimated_randoms = pet.estimate_randoms(delayed_sino)

        # Write estimated randoms to file
        estimated_randoms.write(args.randoms_file)

        # Copy acquisition data into numpy arrays
        prompt_array = prompt_sino.as_array()
        delayed_array = delayed_sino.as_array()
        randoms_array = estimated_randoms.as_array()

        # Print out acquisition data dimensions and statistics
        print(f"Prompt sinogram shape: {prompt_array.shape}")
        print(f"Delayed sinogram shape: {delayed_array.shape}")
        print(f"Estimated randoms shape: {randoms_array.shape}")
        print(f"Total delayed coincidences: {np.sum(delayed_array)}")
        print(f"Total estimated randoms: {np.sum(randoms_array)}")
        print(f"Max delayed value: {np.max(delayed_array)}")
        print(f"Max estimated randoms value: {np.max(randoms_array)}")

        # Compare with original delayed coincidences
        orig_delayed_sino = pet.AcquisitionData(args.delayed_sinogram)
        orig_delayed_array = orig_delayed_sino.as_array()
        print(f"Original delayed sinogram shape: {orig_delayed_array.shape}")
        print(f"Total original delayed coincidences: {np.sum(orig_delayed_array)}")
        print(f"Max original delayed value: {np.max(orig_delayed_array)}")

        # Display a single sinogram if not in non-interactive mode
        if not args.non_interactive:
            slice_idx = prompt_array.shape[0] // 2
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(prompt_array[slice_idx], cmap='hot')
            plt.title('Prompt Sinogram')
            plt.colorbar()
            plt.subplot(1, 3, 2)
            plt.imshow(delayed_array[slice_idx], cmap='hot')
            plt.title('Delayed Sinogram')
            plt.colorbar()
            plt.subplot(1, 3, 3)
            plt.imshow(randoms_array[slice_idx], cmap='hot')
            plt.title('Estimated Randoms')
            plt.colorbar()
            plt.tight_layout()
            plt.show()

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```