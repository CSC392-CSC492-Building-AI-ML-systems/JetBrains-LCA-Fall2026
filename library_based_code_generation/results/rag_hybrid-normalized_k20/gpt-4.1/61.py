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
    parser.add_argument('--run-recon', action='store_true', help='Run the reconstruction if non-cartesian code was compiled.')
    parser.add_argument('--trajectory', type=str, default='radial', choices=['cartesian', 'radial', 'goldenangle', 'grpe'], help='Trajectory type.')
    parser.add_argument('--show-plots', action='store_true', help='Show plots.')
    return parser.parse_args()

def import_engine(engine):
    if engine.lower() == 'gadgetron':
        import sirf.Gadgetron as eng
    elif engine.lower() == 'parallelproj':
        import sirf.Parallelproj as eng
    else:
        raise ValueError(f"Unknown engine: {engine}")
    return eng

def cg_reconstruction(acq_data, eng, trajectory_type, show_plots=False):
    # Compute coil sensitivity maps
    csms = eng.CoilSensitivityData()
    csms.calculate(acq_data)
    # Set up acquisition model
    if hasattr(eng, 'AcquisitionModelUsingParallelproj'):
        acq_model = eng.AcquisitionModelUsingParallelproj(acq_data, csms)
    else:
        acq_model = eng.AcquisitionModel(acq_data, csms)
    # Set trajectory
    if trajectory_type == 'radial':
        acq_data.set_radial2D_trajectory()
    elif trajectory_type == 'goldenangle':
        acq_data.set_data_trajectory('goldenangle')
    elif trajectory_type == 'grpe':
        acq_data.set_grpe_trajectory()
    elif trajectory_type == 'cartesian':
        pass  # Default
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    # Backward projection (adjoint)
    initial_image = acq_model.backward(acq_data)
    # CG iterations
    x = initial_image.clone()
    b = acq_model.backward(acq_data)
    r = b - acq_model.forward(x)
    p = r.clone()
    rsold = (r * r).sum()
    num_iters = 10
    for i in range(num_iters):
        Ap = acq_model.forward(p)
        alpha = rsold / ((Ap * Ap).sum() + 1e-8)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = (r * r).sum()
        if rsnew.sqrt().as_array().max() < 1e-6:
            break
        p = r + p * (rsnew / rsold)
        rsold = rsnew
    if show_plots:
        try:
            import matplotlib.pyplot as plt
            plt.imshow(x.as_array()[0,0,:,:], cmap='gray')
            plt.title('Reconstructed Image')
            plt.show()
        except Exception:
            pass
    return x

def main():
    try:
        args = parse_args()
        eng = import_engine(args.engine)
        # Locate raw data file
        if args.raw_data_file:
            raw_data_file = args.raw_data_file
        else:
            if args.data_path:
                raw_data_file = os.path.join(args.data_path, 'simulated_MR_2D_radial.h5')
            else:
                raw_data_file = os.path.join(eng.examples_data_path(), 'simulated_MR_2D_radial.h5')
        if not os.path.isfile(raw_data_file):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_file}")
        # Read acquisition data
        acq_data = eng.AcquisitionData(raw_data_file)
        # Pre-process acquisition data
        processor = eng.AcquisitionDataProcessor()
        acq_data = processor.preprocess_acquisition_data(acq_data)
        # Set trajectory
        if args.trajectory == 'radial':
            acq_data.set_radial2D_trajectory()
        elif args.trajectory == 'goldenangle':
            acq_data.set_data_trajectory('goldenangle')
        elif args.trajectory == 'grpe':
            acq_data.set_grpe_trajectory()
        elif args.trajectory == 'cartesian':
            pass  # Default
        else:
            raise ValueError(f"Unknown trajectory type: {args.trajectory}")
        # Sort processed acquisition data
        if hasattr(acq_data, 'sort'):
            acq_data.sort()
        # Perform reconstruction
        if args.run_recon:
            recon_image = cg_reconstruction(acq_data, eng, args.trajectory, args.show_plots)
            recon_image.write(args.output_file)
            print(f"Reconstruction complete. Output written to {args.output_file}")
        else:
            print("Reconstruction not run. Use --run-recon to perform reconstruction.")
    except Exception as e:
        print("An error occurred during execution:")
        print(str(e))
        traceback.print_exc()

if __name__ == '__main__':
    main()
```