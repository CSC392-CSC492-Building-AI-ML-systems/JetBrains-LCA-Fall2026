```python
import sys
import os
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative RPE reconstruction with SIRF.')
    parser.add_argument('--raw-data-file', type=str, default=None, help='Raw k-space data file (HDF5).')
    parser.add_argument('--data-path', type=str, default=None, help='Path to data files.')
    parser.add_argument('--output-file', type=str, default='simulated_data.h5', help='Output file for simulated data.')
    parser.add_argument('--engine', type=str, default='Gadgetron', choices=['Gadgetron', 'ISMRMRD'], help='Reconstruction engine.')
    parser.add_argument('--run-recon', action='store_true', help='Run reconstruction if non-cartesian code was compiled.')
    parser.add_argument('--trajectory', type=str, default='radial', choices=['cartesian', 'radial', 'goldenangle', 'grpe'], help='Trajectory type.')
    parser.add_argument('--show-plots', action='store_true', help='Show plots.')
    return parser.parse_args()

def import_engine(engine):
    if engine.lower() == 'gadgetron':
        import sirf.Gadgetron as eng
    elif engine.lower() == 'ismrmrd':
        import sirf.ISMRMRD as eng
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return eng

def cg_reconstruction(acq_data, eng, num_iter=10):
    # Compute coil sensitivity maps
    csm = eng.CoilSensitivityData()
    csm.calculate(acq_data)
    # Set up acquisition model
    acq_model = eng.AcquisitionModel(acq_data, csm)
    # Backward projection (adjoint)
    initial_image = acq_model.backward(acq_data)
    image = initial_image.clone()
    rhs = acq_model.backward(acq_data)
    # Define symmetric operator for CG
    def A(x):
        return acq_model.backward(acq_model.forward(x))
    r = rhs - A(image)
    p = r.clone()
    rsold = np.real((r * r).sum())
    for i in range(num_iter):
        Ap = A(p)
        alpha = rsold / np.real((p * Ap).sum())
        image += p * alpha
        r -= Ap * alpha
        rsnew = np.real((r * r).sum())
        if np.sqrt(rsnew) < 1e-6:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew
    return image

def main():
    args = parse_args()
    try:
        eng = import_engine(args.engine)
        # Locate raw data file
        if args.raw_data_file:
            raw_data_file = args.raw_data_file
        else:
            if args.data_path:
                raw_data_file = os.path.join(args.data_path, 'meas_MID00053_FID00053_gre_radial_v4.h5')
            else:
                raw_data_file = os.path.join(eng.examples_data_path('MR'), 'meas_MID00053_FID00053_gre_radial_v4.h5')
        if not os.path.isfile(raw_data_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
        # Read acquisition data
        acq_data = eng.AcquisitionData(raw_data_file)
        # Pre-process acquisition data
        processor = eng.AcquisitionDataProcessor()
        processed_acq_data = processor.preprocess_acquisition_data(acq_data)
        # Set trajectory
        processed_acq_data.set_data_trajectory(args.trajectory)
        # Sort processed acquisition data
        processed_acq_data.sort()
        # Perform reconstruction if requested
        if args.run_recon:
            image = cg_reconstruction(processed_acq_data, eng)
            image.write(args.output_file)
            if args.show_plots:
                try:
                    import matplotlib.pyplot as plt
                    plt.imshow(np.abs(image.as_array()[0,0,:,:]), cmap='gray')
                    plt.title('Reconstructed Image')
                    plt.show()
                except Exception as e:
                    print(f"Plotting failed: {e}")
        else:
            print("Reconstruction not run. Use --run-recon to perform reconstruction.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
```