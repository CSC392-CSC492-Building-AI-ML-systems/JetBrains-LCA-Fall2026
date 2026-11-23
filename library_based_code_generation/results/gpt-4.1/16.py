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
    parser.add_argument('--transform_type', type=str, choices=['affine', 'rigid', 'none'], default='none', help='Type of transform for attenuation image')
    parser.add_argument('--non_interactive', action='store_true', help='Run in non-interactive mode')
    parser.add_argument('--default_template', type=str, default='template.hs', help='Default template sinogram')
    parser.add_argument('--default_attenuation', type=str, default='atten.hv', help='Default attenuation image')
    parser.add_argument('--default_norm', type=str, default='norm.n', help='Default ECAT8 bin normalisation file')
    args = parser.parse_args()

    template_file = file_or_default(args.template, args.default_template)
    attenuation_file = file_or_default(args.attenuation, args.default_attenuation)
    norm_file = file_or_default(args.norm, args.default_norm)

    if not os.path.isfile(template_file):
        print(f"Template sinogram file not found: {template_file}")
        sys.exit(1)

    acq_data = pet.AcquisitionData(template_file)
    acq_model = pet.AcquisitionModelUsingRayTracingMatrix()
    acq_model.set_up(acq_data, pet.ImageData(attenuation_file) if os.path.isfile(attenuation_file) else acq_data.create_uniform_image(1.0))

    attenuation_model = None
    if os.path.isfile(attenuation_file):
        attn_image = pet.ImageData(attenuation_file)
        if args.transform_type != 'none' and args.transform:
            if args.transform_type == 'affine':
                attn_image = attn_image.transform(args.transform, 'affine')
            elif args.transform_type == 'rigid':
                attn_image = attn_image.transform(args.transform, 'rigid')
        attenuation_model = pet.AcquisitionSensitivityModel(attn_image, acq_model, False)

    norm_model = None
    if os.path.isfile(norm_file):
        norm_model = pet.AcquisitionSensitivityModel(norm_file)

    if norm_model and attenuation_model:
        sens_model = pet.AcquisitionSensitivityModel()
        sens_model.add_sensitivity(norm_model)
        sens_model.add_sensitivity(attenuation_model)
    elif norm_model:
        sens_model = norm_model
    elif attenuation_model:
        sens_model = attenuation_model
    else:
        print("Neither normalisation nor attenuation data found.")
        sys.exit(1)

    sens_model.set_up(acq_data)
    mult_sino = acq_data.clone()
    sens_model.forward(mult_sino)

    mult_sino.write(args.output)
    if not args.non_interactive:
        print(f"Multiplicative sinogram written to {args.output}")

if __name__ == '__main__':
    main()
```