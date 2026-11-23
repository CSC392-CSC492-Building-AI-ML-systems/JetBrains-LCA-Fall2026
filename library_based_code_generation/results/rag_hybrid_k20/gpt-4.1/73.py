```python
import argparse
import sys
import numpy as np
import sirf.STIR as pet
from sirf.Utilities import show_2D_array

def main():
    parser = argparse.ArgumentParser(description='Listmode-to-sinograms conversion using SIRF.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data files')
    parser.add_argument('--listmode-file', type=str, required=True, help='Listmode file')
    parser.add_argument('--output-prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--raw-data-template', type=str, required=True, help='Raw data template')
    parser.add_argument('--time-interval', type=float, nargs=2, metavar=('START', 'END'), default=None, help='Scanning time interval to convert (start end)')
    parser.add_argument('--engine', type=str, default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage-scheme', type=str, default='memory', choices=['memory', 'file'], help='Acquisition data storage scheme')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive mode')
    args = parser.parse_args()

    # Set acquisition data storage scheme
    pet.AcquisitionData.set_storage_scheme(args.storage_scheme)

    # Read acquisition data template
    template_path = args.raw_data_template
    acq_template = pet.AcquisitionData(template_path)

    # Create listmode-to-sinograms converter
    lm2sino = pet.ListmodeToSinograms()
    lm2sino.set_input(args.listmode_file)
    lm2sino.set_output_file_prefix(args.output_prefix)
    lm2sino.set_template_file(template_path)

    # Set time interval if provided
    if args.time_interval is not None:
        lm2sino.set_time_interval(args.time_interval[0], args.time_interval[1])

    # Set flags (example: store prompts and delayed events)
    lm2sino.set_flag('store_prompts', True)
    lm2sino.set_flag('store_delayeds', True)

    # Set up the converter
    lm2sino.set_up()

    # Perform the conversion
    lm2sino.process()

    # Get access to the sinograms
    sino_data = lm2sino.get_output()

    # Copy acquisition data into a Python array
    sino_array = sino_data.as_array()

    # Print acquisition data dimensions
    print('Sinogram data dimensions:', sino_array.shape)

    # Optionally show a 2D array of the acquisition data
    if not args.no_interactive:
        show_2D_array(sino_array[0, :, :], 'Sinogram slice [0,:,:]')

    # Estimate randoms
    randoms = lm2sino.estimate_randoms()
    randoms_array = randoms.as_array()

    # Optionally show a 2D array of the randoms
    if not args.no_interactive:
        show_2D_array(randoms_array[0, :, :], 'Randoms slice [0,:,:]')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
```