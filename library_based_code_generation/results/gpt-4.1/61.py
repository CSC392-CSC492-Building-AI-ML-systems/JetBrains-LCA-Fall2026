```python
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative reconstruction with RPE data using SIRF.')
    parser.add_argument('--raw-data-file', type=str, required=True, help='Raw k-space data file (HDF5).')
    parser.add_argument('--data-path', type=str, default='.', help='Path to data files.')
    parser.add_argument('--simulated-data-out', type=str, default=None, help='Output file for simulated data.')
    parser.add_argument('--engine', type=str, default='Gadgetron', choices=['Gadgetron', 'PETRA'], help='Reconstruction engine.')
    parser.add_argument('--run-recon', action='store_true', help='Run reconstruction if non-cartesian code was compiled.')
    parser.add_argument('--trajectory', type=str, default='radial', choices=['cartesian', 'radial', 'goldenangle', 'grpe'], help='Trajectory type.')
    parser.add_argument('--show-plots', action='store_true', help='Show plots.')
    return parser.parse_args()

def import_engine(engine_name):
    if engine_name.lower() == 'gadgetron':
        import sirf.Gadgetron as engine
    elif engine_name.lower() == 'petra':
        import sirf.PETRA as engine
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    return engine

def cg_reconstruction(engine, acq_data, niter=10):
    # Compute coil sensitivity maps
    csms = engine.CoilSensitivityData()
    csms.calculate(acq_data)
    # Set up acquisition model
    acq_model = engine.AcquisitionModel(acq_data, csms)
    # Define symmetrical operator for CG
    def A(x):
        return acq_model.forward(x)
    def AH(y):
        return acq_model.backward(y)
    # Initial image estimate
    image = engine.ImageData(acq_model.get_image_dimensions())
    image.fill(0)
    # Backward projection of measured data
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
        engine = import_engine(args.engine)
        # Locate raw data file
        raw_data_path = os.path.join(args.data_path, args.raw_data_file)
        if not os.path.isfile(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
        # Read acquisition data
        acq_data = engine.AcquisitionData(raw_data_path)
        # Pre-process acquisition data
        preprocessor = engine.AcquisitionDataProcessor()
        processed_acq_data = preprocessor.process(acq_data)
        # Set trajectory
        if args.trajectory == 'cartesian':
            processed_acq_data.set_trajectory_type(engine.AcquisitionData.TRAJECTORY_CARTESIAN)
        elif args.trajectory == 'radial':
            processed_acq_data.set_trajectory_type(engine.AcquisitionData.TRAJECTORY_RADIAL)
        elif args.trajectory == 'goldenangle':
            processed_acq_data.set_trajectory_type(engine.AcquisitionData.TRAJECTORY_GOLDENANGLE)
        elif args.trajectory == 'grpe':
            processed_acq_data.set_trajectory_type(engine.AcquisitionData.TRAJECTORY_GRPE)
        else:
            raise ValueError(f"Unknown trajectory type: {args.trajectory}")
        # Sort processed acquisition data
        processed_acq_data.sort()
        # Perform reconstruction if requested
        if args.run_recon:
            recon_image = cg_reconstruction(engine, processed_acq_data)
            if args.simulated_data_out:
                recon_image.write(args.simulated_data_out)
            if args.show_plots:
                try:
                    import matplotlib.pyplot as plt
                    plt.imshow(recon_image.as_array()[recon_image.as_array().shape[0]//2,:,:].real, cmap='gray')
                    plt.title('Reconstructed Image (central slice)')
                    plt.show()
                except Exception as e:
                    print(f"Plotting failed: {e}")
        else:
            print("Reconstruction not run. Use --run-recon to perform reconstruction.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
```