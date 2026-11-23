```python
import argparse
import os
import sys
import sirf.STIR as pet

def file_or_default(path, default):
    return path if path and os.path.isfile(path) else default

def main():
    parser = argparse.ArgumentParser(description="Create multiplicative sinograms using SIRF.")
    parser.add_argument('--template', type=str, help='Path to template sinogram file')
    parser.add_argument('--attenuation', type=str, help='Path to attenuation image file')
    parser.add_argument('--norm', type=str, help='Path to ECAT8 bin normalisation file')
    parser.add_argument('--output', type=str, required=True, help='Output file for multiplicative sinogram')
    parser.add_argument('--transform', type=str, help='Path to transform for attenuation image')
    parser.add_argument('--transform-type', type=str, choices=['none', 'rigid', 'affine'], default='none', help='Type of transform for attenuation image')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
    parser.add_argument('--default-template', type=str, default='template.hs', help='Default template sinogram')
    parser.add_argument('--default-attenuation', type=str, default='attenuation.hv', help='Default attenuation image')
    parser.add_argument('--default-norm', type=str, default='norm.n', help='Default normalisation file')
    args = parser.parse_args()

    template_file = file_or_default(args.template, args.default_template)
    attenuation_file = file_or_default(args.attenuation, args.default_attenuation)
    norm_file = file_or_default(args.norm, args.default_norm)

    if not os.path.isfile(template_file):
        print(f"Template sinogram file not found: {template_file}")
        sys.exit(1)

    acq_data = pet.AcquisitionData(template_file)

    attenuation_image = None
    if os.path.isfile(attenuation_file):
        attenuation_image = pet.ImageData(attenuation_file)
        if args.transform and os.path.isfile(args.transform):
            if args.transform_type == 'rigid':
                # Example: apply rigid transform (pseudo-code, replace with actual SIRF transform if available)
                pass
            elif args.transform_type == 'affine':
                # Example: apply affine transform (pseudo-code, replace with actual SIRF transform if available)
                pass
            # else: no transform
    else:
        attenuation_image = None

    norm_present = os.path.isfile(norm_file)
    attenuation_present = attenuation_image is not None

    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(acq_data, attenuation_image if attenuation_present else acq_data.create_uniform_image(1.0))

    sens_model = None
    if norm_present and attenuation_present:
        sens_model = pet.AcquisitionSensitivityModel(norm_file, attenuation_image)
    elif norm_present:
        sens_model = pet.AcquisitionSensitivityModel(norm_file)
    elif attenuation_present:
        sens_model = pet.AcquisitionSensitivityModel(attenuation_image)
    else:
        print("Neither normalisation nor attenuation data provided. Exiting.")
        sys.exit(1)

    sens_model.set_up(acq_data)

    multiplicative_sino = sens_model.get_acquisition_sensitivity(acq_data)

    multiplicative_sino.write(args.output)
    if not args.non_interactive:
        print(f"Multiplicative sinogram written to {args.output}")

if __name__ == '__main__':
    main()
```