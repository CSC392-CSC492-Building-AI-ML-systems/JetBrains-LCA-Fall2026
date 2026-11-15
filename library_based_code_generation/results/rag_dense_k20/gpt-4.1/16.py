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
    parser.add_argument('--transform_type', type=str, choices=['affine', 'rigid', 'none'], default='none', help='Type of transform to apply to attenuation image')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    parser.add_argument('--default_template', type=str, default='template.hs', help='Default template sinogram file')
    parser.add_argument('--default_attenuation', type=str, default='attenuation.nii', help='Default attenuation image file')
    parser.add_argument('--default_norm', type=str, default='norm.n.hdr', help='Default ECAT8 bin normalisation file')
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
        if args.transform and args.transform_type != 'none':
            import sirf.Reg as reg
            tm = reg.AffineTransformation(args.transform) if args.transform_type == 'affine' else reg.RigidTransformation(args.transform)
            resampler = reg.NiftyResample()
            resampler.set_reference_image(attenuation_image)
            resampler.set_floating_image(attenuation_image)
            resampler.set_transformation(tm)
            attenuation_image = resampler.forward(attenuation_image)

    norm_present = os.path.isfile(norm_file)
    attenuation_present = attenuation_image is not None

    if not norm_present and not attenuation_present:
        print("Neither normalisation nor attenuation data provided or found.")
        sys.exit(1)

    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(acq_data, attenuation_image if attenuation_present else acq_data.clone())

    sens_model = None
    if norm_present and attenuation_present:
        sens_model = pet.AcquisitionSensitivityModel(norm_file, attenuation_image)
    elif norm_present:
        sens_model = pet.AcquisitionSensitivityModel(norm_file)
    elif attenuation_present:
        sens_model = pet.AcquisitionSensitivityModel(attenuation_image)
    else:
        print("No sensitivity model could be created.")
        sys.exit(1)

    sens_model.set_up(acq_data)
    multiplicative_sino = acq_data.clone()
    sens_model.forward(multiplicative_sino)

    multiplicative_sino.write(args.output)
    if not args.non_interactive:
        print(f"Multiplicative sinogram written to {args.output}")

if __name__ == '__main__':
    main()
```