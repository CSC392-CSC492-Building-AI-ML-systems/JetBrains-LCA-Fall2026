```python
import blenderproc as bproc
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_file", required=True, help="Path to the .obj file")
    parser.add_argument("--texture_files", nargs='+', required=True, help="Paths to texture files")
    parser.add_argument("--label_mapping", required=True, help="JSON file mapping object names to labels")
    parser.add_argument("--output_dir", required=True, help="Directory to save output .hdf5 file")
    return parser.parse_args()

def main():
    args = parse_args()
    bproc.init()

    # Load the scene from .obj file and textures
    objs = bproc.loader.load_obj(args.obj_file)
    for obj, tex_path in zip(objs, args.texture_files):
        bproc.texture.load_texture(obj, tex_path)

    # Load label mapping
    with open(args.label_mapping, 'r') as f:
        label_mapping = json.load(f)

    # Label objects
    for obj in objs:
        obj_name = obj.get_name()
        label = label_mapping.get(obj_name, "unknown")
        obj.set_cp("category_id", label)

    # Separate walls, floors, ceilings and assign labels
    for obj in objs:
        name = obj.get_name().lower()
        if "wall" in name:
            obj.set_cp("category_id", "wall")
        elif "floor" in name:
            obj.set_cp("category_id", "floor")
        elif "ceiling" in name:
            obj.set_cp("category_id", "ceiling")

    # Make lamp and ceiling objects emit light
    for obj in objs:
        name = obj.get_name().lower()
        if "lamp" in name or "ceiling" in name:
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location(obj.get_location())
            light.set_energy(1000)

    # Create BVH tree
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # Sample camera locations and rotations above the floor
    cam_poses = []
    sampler = bproc.sampler.Front3DPointInRoomSampler(objs)
    for _ in range(20):
        location, rotation = sampler.sample()
        # Ensure camera is above floor
        if location[2] < 1.0:
            continue
        # Check for obstacles in front of camera
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        if not bproc.camera.perform_obstacle_check(cam2world, bvh_tree):
            continue
        # Check scene coverage score
        coverage = bproc.camera.estimate_scene_coverage(cam2world, objs)
        if coverage < 0.2:
            continue
        cam_poses.append(cam2world)

    # Set up camera
    bproc.camera.add_camera_pose(cam_poses)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```