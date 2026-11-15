```python
import argparse
import numpy as np
import sys
import sirf.STIR as pet

def main():
    parser = argparse.ArgumentParser(description='Listmode-to-sinograms conversion using SIRF.')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data files')
    parser.add_argument('--listmode-file', type=str, required=True, help='Listmode file')
    parser.add_argument('--output-prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--raw-data-template', type=str, required=True, help='Raw data template')
    parser.add_argument('--time-interval', type=float, nargs=2, metavar=('START', 'END'), default=[0.0, 0.0], help='Scanning time interval to convert (start end)')
    parser.add_argument('--engine', type=str, default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage-scheme', type=str, default='file', choices=['file', 'memory'], help='Acquisition data storage scheme')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive mode')
    args = parser.parse_args()

    # Set acquisition data storage scheme
    pet.AcquisitionData.set_storage_scheme(args.storage_scheme)

    # Read acquisition data template
    template_path = args.raw_data_template
    acq_template = pet.AcquisitionData(template_path)

    # Create ListmodeToSinograms converter
    lm2sino = pet.ListmodeToSinograms()
    lm2sino.set_input(args.listmode_file)
    lm2sino.set_output_file_prefix(args.output_prefix)
    lm2sino.set_template_file(template_path)

    # Set time interval
    start_time, end_time = args.time_interval
    if start_time != 0.0 or end_time != 0.0:
        lm2sino.set_time_interval(start_time, end_time)

    # Set flags (example: store prompts and delayed events)
    lm2sino.set_flag('store_prompts', True)
    lm2sino.set_flag('store_delayeds', True)

    # Set up converter
    lm2sino.set_up()

    # Perform conversion
    lm2sino.process()

    # Get access to sinograms
    acq_data = lm2sino.get_output()

    # Copy acquisition data into a Python array
    acq_array = acq_data.as_array()

    # Print acquisition data dimensions
    print('Acquisition data dimensions:', acq_array.shape)

    # Optionally show a 2D array of the acquisition data
    if not args.no_interactive:
        import matplotlib.pyplot as plt
        plt.imshow(acq_array[0, :, :], cmap='gray')
        plt.title('Sinogram (first segment)')
        plt.xlabel('Tangential position')
        plt.ylabel('View')
        plt.colorbar()
        plt.show()

    # Estimate randoms
    randoms = pet.estimate_randoms(acq_data)

    # Convert randoms to array
    randoms_array = randoms.as_array()

    # Optionally show a 2D array of the randoms
    if not args.no_interactive:
        plt.imshow(randoms_array[0, :, :], cmap='hot')
        plt.title('Estimated Randoms (first segment)')
        plt.xlabel('Tangential position')
        plt.ylabel('View')
        plt.colorbar()
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
```