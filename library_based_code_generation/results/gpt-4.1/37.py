```python
import argparse
import os
import blenderproc as bproc
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="BlenderProc pipeline for house and chair replacement.")
    parser.add_argument("--house_json", required=True, help="Path to the house.json file.")
    parser.add_argument("--chair_obj", required=True, help="Path to the chair object file.")
    parser.add_argument("--label_map_csv", required=True, help="Path to the label mapping CSV file.")
    parser.add_argument("--output_dir", default="output", help="Output directory for rendered data.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()

    # Load house objects with label mapping
    house_objects = bproc.loader.load_suncg(args.house_json, label_mapping_csv=args.label_map_csv)

    # Load the chair object
    chair_objs = bproc.loader.load_obj(args.chair_obj)
    if not chair_objs:
        raise RuntimeError("Failed to load chair object from {}".format(args.chair_obj))
    chair_obj = chair_objs[0]

    # Find all chair objects in the scene
    chair_class_ids = [obj.get_cp("category_id") for obj in house_objects if obj.get_cp("category") == "chair"]
    chair_objs_in_scene = [obj for obj in house_objects if obj.get_cp("category") == "chair"]

    # Find all floor objects (for collision ignore)
    floor_objs = [obj for obj in house_objects if obj.get_cp("category") == "floor"]

    # Replace all chair objects
    for orig_chair in chair_objs_in_scene:
        # Duplicate the loaded chair object
        new_chair = chair_obj.copy()
        new_chair.set_name("replaced_chair")
        # Copy properties
        for k, v in orig_chair.get_all_cp().items():
            new_chair.set_cp(k, v)
        # Set pose: match location, random z rotation
        location = orig_chair.get_location()
        rotation = orig_chair.get_rotation_euler()
        random_z = random.uniform(0, 2 * np.pi)
        new_chair.set_location(location)
        new_chair.set_rotation_euler([rotation[0], rotation[1], random_z])
        # Ignore collision with floors
        for floor in floor_objs:
            bproc.object.disable_collision_between(new_chair, floor)
        # Add to scene
        bproc.scene.add_object(new_chair)
        # Remove original chair
        orig_chair.delete()

    # Filter out invalid objects
    bproc.object.delete_invalid()

    # Make all Suncg objects emit light
    for obj in bproc.scene.get_objects():
        if obj.get_cp("suncg") is not None:
            bproc.material.make_emissive(obj, emission_strength=5.0)

    # Initialize point sampler and BVH tree
    point_sampler = bproc.sampler.SuncgRoomPointSampler(house_json_path=args.house_json)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.scene.get_objects())

    # Sample camera poses
    camera_poses = []
    num_cameras = 10
    for _ in range(num_cameras):
        for _ in range(100):  # Try up to 100 times to find a valid pose
            location = point_sampler.sample()
            # Ensure at least 1m from obstacles
            min_dist = bvh_tree.compute_min_distance(location)
            if min_dist < 1.0:
                continue
            # Look at random point in the house
            target = point_sampler.sample()
            cam2world_matrix = bproc.camera.camera_pose_from_lookat(location, target)
            # Check if view covers at least 40% of the scene
            coverage = bproc.camera.coverage(cam2world_matrix, bproc.scene.get_objects())
            if coverage < 0.4:
                continue
            camera_poses.append(cam2world_matrix)
            break

    # Add cameras to the scene
    bproc.camera.add_camera_pose(camera_poses)

    # Enable normal, depth, segmentation rendering, and alpha channel
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.enable_alpha_channel()

    # Render the scene
    data = bproc.renderer.render()

    # Write to .hdf5 file
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```