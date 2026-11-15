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
    parser.add_argument('--storage-scheme', type=str, default='memory', choices=['memory', 'file'], help='Acquisition data storage scheme')
    parser.add_argument('--no-interactive', action='store_true', help='Disable interactive mode')
    args = parser.parse_args()

    pet.AcquisitionData.set_storage_scheme(args.storage_scheme)

    acq_data_template = pet.AcquisitionData(args.raw_data_template)

    lm2sino = pet.ListmodeToSinograms()
    lm2sino.set_input(args.listmode_file)
    lm2sino.set_output_file_prefix(args.output_prefix)
    lm2sino.set_template_file(args.raw_data_template)
    lm2sino.set_time_interval(*args.time_interval)
    lm2sino.set_flag('store_prompts', True)
    lm2sino.set_flag('store_delayeds', True)
    lm2sino.set_flag('store_singles', True)
    lm2sino.set_flag('write_interval_data', False)
    lm2sino.set_flag('write_listmode_events', False)
    lm2sino.set_flag('interactive', not args.no_interactive)

    lm2sino.set_up()
    lm2sino.process()

    acq_data = lm2sino.get_output()
    acq_array = acq_data.as_array()
    print('Acquisition data dimensions:', acq_array.shape)

    if not args.no_interactive:
        import matplotlib.pyplot as plt
        plt.imshow(acq_array[0, :, :], cmap='gray')
        plt.title('Sinogram (first segment)')
        plt.show()

    randoms = lm2sino.estimate_randoms()
    randoms_array = randoms.as_array()
    if not args.no_interactive:
        import matplotlib.pyplot as plt
        plt.imshow(randoms_array[0, :, :], cmap='hot')
        plt.title('Estimated Randoms (first segment)')
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
```