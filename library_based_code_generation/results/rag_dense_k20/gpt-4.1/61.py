```python
import argparse
import sys
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative reconstruction with RPE data using SIRF.')
    parser.add_argument('--raw-data-file', type=str, required=True, help='Raw k-space data file (HDF5).')
    parser.add_argument('--data-path', type=str, default='.', help='Path to data files.')
    parser.add_argument('--output-file', type=str, default='simulated_data.h5', help='Output file for simulated data.')
    parser.add_argument('--engine', type=str, choices=['Gadgetron', 'Parallelproj'], default='Gadgetron', help='Reconstruction engine.')
    parser.add_argument('--run-recon', action='store_true', help='Run reconstruction if non-cartesian code was compiled.')
    parser.add_argument('--trajectory', type=str, choices=['cartesian', 'radial', 'goldenangle', 'grpe'], default='radial', help='Trajectory type.')
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

def define_sym_op(acq_model):
    # For CG, we need a symmetric operator: A^H A
    class SymOp:
        def __init__(self, acq_model):
            self.acq_model = acq_model
        def __call__(self, x):
            return self.acq_model.backward(self.acq_model.forward(x))
    return SymOp(acq_model)

def conjugate_gradient(sym_op, rhs, x0, niter=10, tol=1e-6):
    x = x0.copy()
    r = rhs - sym_op(x)
    p = r.copy()
    rsold = np.real((r * r.conjugate()).sum())
    for i in range(niter):
        Ap = sym_op(p)
        alpha = rsold / np.real((p * Ap.conjugate()).sum())
        x += alpha * p
        r -= alpha * Ap
        rsnew = np.real((r * r.conjugate()).sum())
        if np.sqrt(rsnew) < tol:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def perform_reconstruction(eng, acq_data, trajectory_type, show_plots):
    try:
        # Preprocess acquisition data
        processed_acq_data = eng.preprocess_acquisition_data(acq_data)
        # Set trajectory
        if trajectory_type == 'radial':
            eng.set_radial2D_trajectory(processed_acq_data)
        elif trajectory_type == 'goldenangle':
            eng.set_goldenangle2D_trajectory(processed_acq_data)
        elif trajectory_type == 'grpe':
            eng.set_grpe_trajectory(processed_acq_data)
        elif trajectory_type == 'cartesian':
            eng.set_data_trajectory(processed_acq_data)
        else:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")
        # Sort processed acquisition data
        processed_acq_data.sort()
        # Compute coil sensitivity maps
        coil_images = eng.CoilImagesData(processed_acq_data)
        # Set up acquisition model
        acq_model = eng.AcquisitionModelUsingParallelproj(processed_acq_data, coil_images)
        acq_model.set_coil_sensitivity_maps(coil_images)
        # Backward projection (rhs)
        rhs = acq_model.backward(processed_acq_data)
        # Initial guess
        x0 = rhs * 0
        # Define symmetric operator
        sym_op = define_sym_op(acq_model)
        # Perform CG
        recon = conjugate_gradient(sym_op, rhs, x0, niter=10)
        if show_plots:
            import matplotlib.pyplot as plt
            plt.imshow(np.abs(recon.as_array()[0,0]), cmap='gray')
            plt.title('Reconstructed Image')
            plt.show()
        return recon
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        sys.exit(1)

def main():
    args = parse_args()
    try:
        eng = import_engine(args.engine)
        raw_data_path = os.path.join(args.data_path, args.raw_data_file)
        if not os.path.isfile(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        acq_data = eng.AcquisitionData(raw_data_path)
        if args.run_recon:
            recon = perform_reconstruction(eng, acq_data, args.trajectory, args.show_plots)
            recon.write(args.output_file)
            print(f"Reconstruction complete. Output written to {args.output_file}")
        else:
            print("Reconstruction not run. Use --run-recon to perform reconstruction.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
```