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
    return parser.parse_args()

def main():
    args = parse_args()
    bproc.init()

    # Load label mapping
    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    # Load the .obj file and textures
    objs = bproc.loader.create_mesh_objects_from_file(args.obj_file, use_texture=True, texture_folder=args.texture_dir)

    # Label objects based on mapping
    for obj in objs:
        obj_name = obj.get_name()
        label = label_map.get(obj_name, "unknown")
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
            bproc.light.light_suncg_scene([obj], emission_strength=10.0)

    # Create BVH tree for all objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # Sample camera locations and rotations above the floor
    camera_poses = []
    min_floor_z = min([bproc.object.min_and_max_point(f)[0][2] for f in floor_objs]) if floor_objs else 0.0
    for _ in range(args.num_cameras * 5):  # Oversample to filter later
        # Sample random location above the floor
        loc = np.array([
            np.random.uniform(-2, 2),
            np.random.uniform(-2, 2),
            min_floor_z + np.random.uniform(1.2, 1.8)
        ])
        # Look at a random point in the scene
        target = np.array([
            np.random.uniform(-1, 1),
            np.random.uniform(-1, 1),
            min_floor_z + np.random.uniform(0.8, 1.2)
        ])
        rot = bproc.camera.rotation_from_forward_vec(target - loc, inplane_rot=np.random.uniform(-np.pi, np.pi))
        # Check for obstacles in front of camera
        ray_origin = loc
        ray_dir = (target - loc) / np.linalg.norm(target - loc)
        hit, _, _, _, _ = bproc.object.scene_ray_cast(ray_origin, ray_dir, distance=2.0)
        if not hit:
            # Check scene coverage score
            score = bproc.camera.scene_coverage_score([loc], [rot], objs)
            if score > 0.2:  # Threshold for coverage
                camera_poses.append((loc, rot))
        if len(camera_poses) >= args.num_cameras:
            break

    # Set up camera intrinsics
    bproc.camera.set_resolution(640, 480)
    bproc.camera.set_intrinsics_from_blender_params(fx=600, fy=600, cx=320, cy=240)

    # Add cameras to the scene
    for loc, rot in camera_poses:
        bproc.camera.add_camera_pose(np.eye(4, 4))
        bproc.camera.set_camera_pose(bproc.math.build_transformation_mat(loc, rot))

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