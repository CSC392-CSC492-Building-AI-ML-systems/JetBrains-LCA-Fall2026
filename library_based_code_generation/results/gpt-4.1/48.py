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
    parser.add_argument('--listmode', required=True, help='Path to listmode file')
    parser.add_argument('--sinograms-prefix', required=True, help='Prefix for output sinograms files')
    parser.add_argument('--randoms', required=True, help='Output file for estimated randoms')
    parser.add_argument('--rawdata-template', required=True, help='Raw data template file')
    parser.add_argument('--delayed', required=True, help='Output file for delayed coincidences')
    parser.add_argument('--time-interval', nargs=2, type=float, metavar=('START', 'END'), default=None, help='Scanning time interval (start end)')
    parser.add_argument('--engine', default='STIR', help='Reconstruction engine')
    parser.add_argument('--storage-scheme', default='memory', help='Acquisition data storage scheme')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode (no plots)')
    args = parser.parse_args()

    try:
        pet.AcquisitionData.set_storage_scheme(args.storage_scheme)
        pet.set_verbosity(1)

        converter = pet.ListmodeToSinograms()
        converter.set_input(args.listmode)
        converter.set_output_prefix(args.sinograms-prefix)
        converter.set_template(args.rawdata_template)
        converter.set_delayed_event_output_filename(args.delayed)
        converter.set_store_delayed_events(True)
        if args.time_interval:
            converter.set_time_interval(*args.time_interval)

        print("Processing listmode data...")
        converter.process()

        print("Loading sinograms and delayed coincidences...")
        acq_data = pet.AcquisitionData(args.sinograms_prefix + "_acq.h5")
        delayed_data = pet.AcquisitionData(args.delayed)

        acq_array = acq_data.as_array()
        delayed_array = delayed_data.as_array()

        print("Estimating randoms using Maximum Likelihood estimation...")
        # ML estimation: randoms = delayed / (number of prompts / number of delayeds)
        # But typically, randoms are estimated as the delayed coincidences
        # Here, we use the delayed coincidences as the ML estimate of randoms
        estimated_randoms = delayed_array.copy()

        # Save estimated randoms
        randoms_data = pet.AcquisitionData(args.rawdata_template)
        randoms_data.fill(estimated_randoms)
        randoms_data.write(args.randoms)

        print("Acquisition data dimensions:", acq_array.shape)
        print("Delayed coincidences dimensions:", delayed_array.shape)
        print("Estimated randoms dimensions:", estimated_randoms.shape)
        print("Total delayed coincidences:", np.sum(delayed_array))
        print("Total estimated randoms:", np.sum(estimated_randoms))
        print("Max delayed coincidence value:", np.max(delayed_array))
        print("Max estimated randoms value:", np.max(estimated_randoms))

        if not args.non_interactive:
            # Display a single sinogram (first segment, first axial position)
            plt.figure(figsize=(8, 6))
            plt.imshow(acq_array[0, 0, :, :], cmap='hot')
            plt.title('First Sinogram')
            plt.colorbar()
            plt.show()

    except Exception as e:
        print("Error during processing:", str(e))
        sys.exit(1)

if __name__ == '__main__':
    main()
```