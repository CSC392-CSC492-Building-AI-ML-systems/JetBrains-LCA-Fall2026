```python
import argparse
import blenderproc as bproc
import os

parser = argparse.ArgumentParser()
parser.add_argument("--blend_file", type=str, required=True, help="Path to the .blend file")
parser.add_argument("--haven_dir", type=str, required=True, help="Path to the haven HDRI directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for rendered data")
args = parser.parse_args()

bproc.init()

bproc.loader.load_blend(args.blend_file)

hdri_path = bproc.haven.get_random_world_background_hdr_img_path_from_haven(args.haven_dir)
bproc.world.set_world_background_hdr_img(hdri_path)

light = bproc.types.Light()
light.set_type("POINT")
light.set_location([2, 2, 3])
light.set_energy(1000)

poi = bproc.object.compute_poi()
cam_poses = bproc.camera.sample_poses_around_point(poi, number_of_poses=5, radius_min=2.0, radius_max=3.0)

bproc.camera.add_camera_pose(cam_poses)

bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output()

data = bproc.renderer.render()

output_path = os.path.join(args.output_dir, "output.hdf5")
bproc.writer.write_hdf5(output_path, data)
```