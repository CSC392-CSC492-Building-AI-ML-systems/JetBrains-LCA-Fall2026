```python
import blenderproc as bproc
import argparse
import os
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_file", required=True, help="Path to the .obj file")
    parser.add_argument("--texture_dir", required=True, help="Directory containing texture files")
    parser.add_argument("--label_map", required=True, help="JSON file mapping object names to labels")
    parser.add_argument("--output_dir", required=True, help="Directory to write output .hdf5 file")
    parser.add_argument("--num_cameras", type=int, default=10, help="Number of camera poses to sample")
    parser.add_argument("--min_coverage", type=float, default=0.5, help="Minimum scene coverage score")
    return parser.parse_args()

def main():
    args = parse_args()
    bproc.init()

    # Load label mapping
    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    # Load the scene from .obj file and textures
    objs = bproc.loader.load_obj(args.obj_file, texture_path=args.texture_dir)

    # Label objects based on mapping
    for obj in objs:
        name = obj.get_name()
        label = label_map.get(name, "unknown")
        obj.set_cp("category_id", label)

    # Separate walls, floors, ceilings and assign labels
    wall_objs, floor_objs, ceil_objs = [], [], []
    for obj in objs:
        name = obj.get_name().lower()
        if "wall" in name:
            obj.set_cp("category_id", "wall")
            wall_objs.append(obj)
        elif "floor" in name:
            obj.set_cp("category_id", "floor")
            floor_objs.append(obj)
        elif "ceiling" in name or "ceiling" in label_map.get(obj.get_name(), "").lower():
            obj.set_cp("category_id", "ceiling")
            ceil_objs.append(obj)

    # Make lamp and ceiling objects emit light
    for obj in objs:
        name = obj.get_name().lower()
        if "lamp" in name or "light" in name or obj in ceil_objs:
            light = bproc.types.Light()
            light.set_type("POINT")
            light.set_location(obj.get_location())
            light.set_energy(1000)

    # Create BVH tree for all objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # Sample camera locations and rotations above the floor
    cam_poses = []
    sampler = bproc.camera.Front3DPointInRoomSampler(
        floor_objs=floor_objs,
        wall_objs=wall_objs,
        ceil_objs=ceil_objs,
        min_height_above_floor=1.0,
        max_height_above_floor=2.0
    )
    num_sampled = 0
    max_trials = args.num_cameras * 10
    trials = 0
    while num_sampled < args.num_cameras and trials < max_trials:
        location, rotation = sampler.sample()
        # Check for obstacles in front of camera
        cam_forward = bproc.math.rotation_matrix_to_forward_vector(rotation)
        ray_start = location
        ray_end = location + cam_forward * 2.0
        hit, _, _, _ = bvh_tree.ray_cast(ray_start, ray_end)
        if not hit:
            # Compute scene coverage score (dummy implementation)
            coverage = np.random.uniform(0.5, 1.0)  # Replace with actual coverage computation
            if coverage >= args.min_coverage:
                bproc.camera.add_camera_pose(np.eye(4, dtype=np.float32))
                bproc.camera.get_camera().set_location(location)
                bproc.camera.get_camera().set_rotation_mat(rotation)
                cam_poses.append((location, rotation))
                num_sampled += 1
        trials += 1

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "output.hdf5")
    bproc.writer.write_hdf5(output_path, data)

if __name__ == "__main__":
    main()
```