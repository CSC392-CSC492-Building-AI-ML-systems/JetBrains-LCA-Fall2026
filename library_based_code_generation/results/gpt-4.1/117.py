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
    parser.add_argument("--output_dir", required=True, help="Directory to save output .hdf5 file")
    parser.add_argument("--num_cameras", type=int, default=10, help="Number of camera poses to sample")
    parser.add_argument("--min_coverage", type=float, default=0.5, help="Minimum scene coverage score")
    return parser.parse_args()

def main():
    args = parse_args()
    bproc.init()

    # Load label mapping
    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    # Load the .obj file with textures
    objs = bproc.loader.load_obj(args.obj_file, texture_path=args.texture_dir)

    # Label objects based on mapping
    for obj in objs:
        name = obj.get_name().lower()
        label = label_map.get(name, "other")
        obj.set_cp("category_id", label)

    # Separate walls, floors, ceilings
    wall_objs, floor_objs, ceil_objs = [], [], []
    for obj in objs:
        name = obj.get_name().lower()
        if "wall" in name:
            wall_objs.append(obj)
            obj.set_cp("category_id", "wall")
        elif "floor" in name:
            floor_objs.append(obj)
            obj.set_cp("category_id", "floor")
        elif "ceiling" in name or "ceil" in name:
            ceil_objs.append(obj)
            obj.set_cp("category_id", "ceiling")

    # Make lamp and ceiling objects emit light
    for obj in objs:
        name = obj.get_name().lower()
        if "lamp" in name or "light" in name or "ceiling" in name or "ceil" in name:
            bproc.types.LightLightSource.create(obj, intensity=1000)

    # Create BVH tree
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # Sample camera poses
    camera_poses = []
    coverage = 0.0
    attempts = 0
    max_attempts = args.num_cameras * 20
    while len(camera_poses) < args.num_cameras and attempts < max_attempts:
        attempts += 1
        # Sample a point above the floor
        if floor_objs:
            floor_obj = floor_objs[0]
            bbox = floor_obj.get_bound_box()
            min_x, min_y, min_z = np.min(bbox, axis=0)
            max_x, max_y, max_z = np.max(bbox, axis=0)
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            z = max_z + np.random.uniform(1.2, 1.8)  # 1.2-1.8m above floor
        else:
            x, y, z = np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(1.2, 1.8)

        location = np.array([x, y, z])
        # Look at the center of the scene
        center = np.array([0, 0, z - 1])
        rotation = bproc.camera.rotation_from_forward_vec(center - location, inplane_rot=np.random.uniform(-np.pi, np.pi))
        cam2world = bproc.math.build_transformation_mat(location, rotation)

        # Check for obstacles in front of camera
        ray_start = location
        ray_end = location + (center - location) / np.linalg.norm(center - location) * 2.0
        hit, _, _, _ = bvh_tree.ray_cast(ray_start, ray_end)
        if hit:
            continue

        # Check scene coverage
        coverage = bproc.camera.visibility_score(cam2world, objs)
        if coverage < args.min_coverage:
            continue

        camera_poses.append(cam2world)

    # Set up camera intrinsics
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(fov=60)

    # Add cameras
    for pose in camera_poses:
        bproc.camera.add_camera_pose(pose)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render
    data = bproc.renderer.render()

    # Write to .hdf5
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```