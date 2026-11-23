```python
import sys
import argparse
import numpy as np
import sirf.STIR as pet
import sirf.Utilities as ut

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

    ut.set_acquisition_data_storage_scheme(args.storage_scheme)

    template_path = ut.examples_data_path(args.data_path, args.raw_data_template)
    acq_template = pet.AcquisitionData(template_path)

    converter = pet.ListmodeToSinograms()
    converter.set_input(ut.examples_data_path(args.data_path, args.listmode_file))
    converter.set_output_file_prefix(args.output_prefix)
    converter.set_template_file(template_path)
    if args.time_interval != [0.0, 0.0]:
        converter.set_time_interval(*args.time_interval)
    converter.set_flag('store_prompts', True)
    converter.set_flag('store_delayeds', True)
    converter.set_flag('store_singles', False)
    converter.set_flag('write_lm_to_projdata', True)
    converter.set_up()
    converter.process()

    sinograms = converter.get_output()
    sino_array = sinograms.as_array()
    print('Acquisition data dimensions:', sino_array.shape)

    if not args.no_interactive:
        ut.show_2D_array('Sinograms (slice 0)', sino_array[0])

    randoms = converter.estimate_randoms()
    randoms_array = randoms.as_array()
    if not args.no_interactive:
        ut.show_2D_array('Randoms (slice 0)', randoms_array[0])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('Error:', e)
        sys.exit(1)
```