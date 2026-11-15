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
    return parser.parse_args()

def main():
    args = parse_args()
    bproc.init()

    # Load label mapping
    with open(args.label_map, "r") as f:
        label_map = json.load(f)

    # Load the scene from .obj file and apply textures
    objs = bproc.loader.load_obj(args.obj_file)
    for obj in objs:
        obj_name = obj.get_name().lower()
        # Assign label from mapping
        label = label_map.get(obj_name, "unknown")
        obj.set_cp("category_id", label)
        # Try to assign texture if available
        tex_path = os.path.join(args.texture_dir, f"{obj_name}.jpg")
        if os.path.exists(tex_path):
            bproc.texture.assign_texture(obj, tex_path)

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
            bproc.light.light_surface(obj, emission_strength=10.0)

    # Create BVH tree for all objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objs)

    # Sample camera poses above the floor
    camera_poses = []
    sampler = bproc.camera.Front3DPointInRoomSampler(
        objects=objs,
        bvh_tree=bvh_tree,
        min_distance=1.0,
        max_distance=5.0,
        min_height_above_floor=1.0,
        max_height_above_floor=2.0
    )
    for _ in range(args.num_cameras * 2):  # Oversample to filter later
        pose = sampler.sample()
        if pose is None:
            continue
        # Check for obstacles in front of camera
        cam_pos = pose[:3, 3]
        cam_dir = pose[:3, 2]
        ray_start = cam_pos
        ray_end = cam_pos + cam_dir * 2.0
        hit, _, _, _ = bvh_tree.ray_cast(ray_start, ray_end)
        if not hit:
            # Compute scene coverage score (dummy example: random for now)
            coverage_score = np.random.uniform(0.5, 1.0)
            if coverage_score > 0.6:
                camera_poses.append(pose)
        if len(camera_poses) >= args.num_cameras:
            break

    # Set camera poses
    bproc.camera.add_camera_pose(np.stack(camera_poses))

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    data = bproc.renderer.render()

    # Write to .hdf5 file
    output_path = os.path.join(args.output_dir, "output.hdf5")
    bproc.writer.write_hdf5(output_path, data)

if __name__ == "__main__":
    main()
```