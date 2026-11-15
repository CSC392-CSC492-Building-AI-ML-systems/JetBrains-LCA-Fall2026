```python
import argparse
import numpy as np
import sys
import traceback

from sirf.STIR import (
    ListmodeToSinograms,
    AcquisitionData,
    estimate_randoms
)
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='Estimate randoms from listmode and compare with delayed coincidences.')
    parser.add_argument('--listmode_file', required=True, help='Path to the listmode file')
    parser.add_argument('--sinograms_prefix', required=True, help='Prefix for output sinograms files')
    parser.add_argument('--randoms_file', required=True, help='File to write estimated randoms')
    parser.add_argument('--raw_data_template', required=True, help='Raw data template file')
    parser.add_argument('--data_path', required=True, help='Path to data files')
    parser.add_argument('--time_interval', nargs=2, type=float, metavar=('START', 'END'), default=None, help='Scanning time interval (start end)')
    parser.add_argument('--recon_engine', default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage_scheme', default='file', choices=['file', 'memory'], help='Acquisition data storage scheme')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode (no plots)')
    args = parser.parse_args()

    try:
        # Set up ListmodeToSinograms converter
        lm2sino = ListmodeToSinograms()
        lm2sino.set_input(args.listmode_file)
        lm2sino.set_output_prefix(args.sinograms_prefix)
        lm2sino.set_template(args.raw_data_template)
        if args.time_interval:
            lm2sino.set_time_interval(args.time_interval[0], args.time_interval[1])
        lm2sino.set_store_delayed(True)
        lm2sino.set_storage_scheme(args.storage_scheme)
        # Process the data
        print('Processing listmode data...')
        lm2sino.process()
        # Get sinograms and delayed coincidences
        print('Loading acquisition data...')
        acq_data = AcquisitionData(args.sinograms_prefix + '_prompts.hs')
        delayed_data = AcquisitionData(args.sinograms_prefix + '_delayeds.hs')
        # Estimate randoms from delayeds
        print('Estimating randoms...')
        estimated_randoms = estimate_randoms(delayed_data)
        estimated_randoms.write(args.randoms_file)
        # Copy data to numpy arrays
        acq_array = acq_data.as_array()
        delayed_array = delayed_data.as_array()
        randoms_array = estimated_randoms.as_array()
        # Print dimensions and statistics
        print('Acquisition data dimensions:', acq_array.shape)
        print('Delayed coincidences dimensions:', delayed_array.shape)
        print('Estimated randoms dimensions:', randoms_array.shape)
        print('Total delayed coincidences:', np.sum(delayed_array))
        print('Total estimated randoms:', np.sum(randoms_array))
        print('Max delayed coincidences:', np.max(delayed_array))
        print('Max estimated randoms:', np.max(randoms_array))
        # Display a single sinogram if not in non-interactive mode
        if not args.non_interactive:
            plt.figure(figsize=(10,4))
            plt.subplot(1,3,1)
            plt.imshow(acq_array[0], cmap='hot')
            plt.title('Prompts Sinogram')
            plt.colorbar()
            plt.subplot(1,3,2)
            plt.imshow(delayed_array[0], cmap='hot')
            plt.title('Delayeds Sinogram')
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.imshow(randoms_array[0], cmap='hot')
            plt.title('Estimated Randoms')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print('Error occurred:', str(e))
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
```