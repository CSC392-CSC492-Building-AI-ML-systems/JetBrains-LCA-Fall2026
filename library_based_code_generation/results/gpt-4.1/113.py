```python
import argparse
import os
import random
import blenderproc as bproc

def main():
    parser = argparse.ArgumentParser(description="BlenderProc pipeline")
    parser.add_argument("--blend_file", type=str, required=True, help="Path to the .blend file")
    parser.add_argument("--haven_dir", type=str, required=True, help="Path to the HDRI haven directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    bproc.init()

    bproc.loader.load_blend(args.blend_file)

    hdri_files = [os.path.join(args.haven_dir, f) for f in os.listdir(args.haven_dir) if f.lower().endswith(('.hdr', '.exr'))]
    if not hdri_files:
        raise RuntimeError("No HDRI files found in the haven directory.")
    hdri_path = random.choice(hdri_files)
    bproc.world.set_world_background_hdr_img(hdri_path)

    light = bproc.types.Light()
    light.set_type("POINT")
    light.set_location([2, 2, 5])
    light.set_energy(1000)

    poi = bproc.object.compute_poi(bproc.scene.get_objects())
    cam_poses = bproc.camera.sample_poses_around_point(poi, number_of_poses=5, distance_range=[2.0, 3.0])

    bproc.camera.add_camera_pose(cam_poses)

    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()

    data = bproc.renderer.render()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "output.hdf5")
    bproc.writer.write_hdf5(output_path, data)

if __name__ == "__main__":
    main()
```