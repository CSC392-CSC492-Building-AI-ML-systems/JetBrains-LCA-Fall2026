```python
import sys
import os
import argparse
import traceback

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative reconstruction with RPE data using SIRF.')
    parser.add_argument('--raw-data-file', type=str, default=None, help='Raw k-space data file (HDF5).')
    parser.add_argument('--data-path', type=str, default=None, help='Path to data files.')
    parser.add_argument('--output-file', type=str, default='simulated_data.h5', help='Output file for simulated data.')
    parser.add_argument('--engine', type=str, default='Gadgetron', choices=['Gadgetron', 'Parallelproj'], help='Reconstruction engine.')
    parser.add_argument('--run-recon', action='store_true', help='Run reconstruction if non-cartesian code was compiled.')
    parser.add_argument('--trajectory', type=str, default='radial', choices=['cartesian', 'radial', 'goldenangle', 'grpe'], help='Trajectory type.')
    parser.add_argument('--show-plots', action='store_true', help='Show plots.')
    return parser.parse_args()

def import_engine(engine):
    if engine == 'Gadgetron':
        import sirf.Gadgetron as eng
    elif engine == 'Parallelproj':
        import sirf.Parallelproj as eng
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return eng

def cg_reconstruction(acq_data, eng, niter=10):
    # Compute coil sensitivity maps
    csm = eng.CoilSensitivityData()
    csm.calculate(acq_data)
    # Set up acquisition model
    acq_model = eng.AcquisitionModel(acq_data, csm)
    # Define symmetrical operator for CG
    def A(x):
        return acq_model.forward(x)
    def AH(y):
        return acq_model.backward(y)
    # Initial image estimate
    image = eng.ImageData(acq_model.get_image_dimensions())
    image.fill(0)
    # Backward projection of data
    rhs = AH(acq_data)
    # CG iterations
    x = image.clone()
    r = rhs - AH(A(x))
    p = r.clone()
    rsold = (r * r).sum()
    for i in range(niter):
        Ap = AH(A(p))
        alpha = rsold / (p * Ap).sum()
        x += p * alpha
        r -= Ap * alpha
        rsnew = (r * r).sum()
        if rsnew.sqrt() < 1e-6:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew
    return x

def main():
    args = parse_args()
    try:
        eng = import_engine(args.engine)
        # Locate raw data file
        if args.raw_data_file:
            raw_data_file = args.raw_data_file
        else:
            if args.data_path:
                raw_data_file = os.path.join(args.data_path, 'meas_MID00053_FID00053_gre_radial_v1.h5')
            else:
                from sirf import examples_data_path
                raw_data_file = os.path.join(examples_data_path('MR'), 'meas_MID00053_FID00053_gre_radial_v1.h5')
        if not os.path.isfile(raw_data_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
        # Read acquisition data
        acq_data = eng.AcquisitionData(raw_data_file)
        # Pre-process acquisition data
        processor = eng.AcquisitionDataProcessor()
        acq_data = processor.preprocess_acquisition_data(acq_data)
        # Set trajectory
        if args.trajectory == 'cartesian':
            pass  # Default
        elif args.trajectory == 'radial':
            acq_data.set_radial2D_trajectory()
        elif args.trajectory == 'goldenangle':
            acq_data.set_radial2D_trajectory(golden_angle=True)
        elif args.trajectory == 'grpe':
            acq_data.set_grpe_trajectory()
        else:
            raise ValueError(f"Unknown trajectory: {args.trajectory}")
        # Sort processed acquisition data
        acq_data.sort()
        # Perform reconstruction if requested
        if args.run_recon:
            image = cg_reconstruction(acq_data, eng)
            image.write(args.output_file)
            if args.show_plots:
                try:
                    import matplotlib.pyplot as plt
                    plt.imshow(image.as_array()[0,0,:,:], cmap='gray')
                    plt.title('Reconstructed Image')
                    plt.show()
                except Exception:
                    print("Plotting failed. Matplotlib may not be installed.")
        else:
            print("Reconstruction not run. Use --run-recon to perform reconstruction.")
    except Exception as e:
        print("An error occurred during execution:")
        print(str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```