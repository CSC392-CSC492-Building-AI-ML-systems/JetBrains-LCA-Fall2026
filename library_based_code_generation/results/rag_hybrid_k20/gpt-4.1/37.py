```python
import argparse
import os
import blenderproc as bproc
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--house_json", required=True, help="Path to house.json file")
    parser.add_argument("--chair_obj", required=True, help="Path to chair object file (.obj/.ply/.blend)")
    parser.add_argument("--label_map_csv", required=True, help="Path to label mapping csv file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()

    # 3. Load house objects with label mapping
    house_objects = bproc.loader.create_mesh_objects_from_file(
        args.house_json,
        label_mapping=args.label_map_csv
    )

    # 4. Load chair object and replace all chairs
    chair_obj = bproc.loader.load_and_create(args.chair_obj)[0]
    all_objs = bproc.object.get_all_blender_mesh_objects()
    chair_objs = [obj for obj in all_objs if "chair" in obj.get_cp("category_id", "").lower() or "chair" in obj.get_name().lower()]
    floor_objs = [obj for obj in all_objs if "floor" in obj.get_cp("category_id", "").lower() or "floor" in obj.get_name().lower()]

    for orig_chair in chair_objs:
        # Duplicate the chair object
        new_chair = chair_obj.duplicate()
        # Copy properties
        for k, v in orig_chair.get_properties().items():
            new_chair.set_cp(k, v)
        # Set pose: copy location, random z rotation
        pose = orig_chair.get_local2world_mat()
        rot_angle = random.uniform(0, 2 * np.pi)
        rot_mat = bproc.math.build_transformation_mat([0,0,0], [0,0,rot_angle])
        pose = pose @ rot_mat
        new_chair.set_local2world_mat(pose)
        # Ignore collision with floors
        for floor in floor_objs:
            bproc.object.disable_collision_between(new_chair, floor)
        # Remove original chair
        orig_chair.delete()

    # 5. Filter out invalid objects
    valid_objs = bproc.object.filter_invalid_objects(bproc.object.get_all_blender_mesh_objects())

    # 6. Make all Suncg objects emit light
    for obj in valid_objs:
        if obj.get_cp("category_id", "").startswith("suncg"):
            obj.make_emissive(emission_strength=5.0)

    # 7. Initialize point sampler and BVH tree
    sampler = bproc.sampler.SuncgPointInRoomSampler(house_json_path=args.house_json)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(valid_objs)

    # 8. Sample camera poses
    cam_poses = []
    num_cams = 5
    for _ in range(num_cams):
        for _ in range(100):  # Try up to 100 times to find a valid pose
            location = sampler.sample()
            # Ensure at least 1m from obstacles
            min_dist = bvh_tree.compute_min_distance(location)
            if min_dist < 1.0:
                continue
            # Random look-at direction
            look_at = location + np.array([random.uniform(-2,2), random.uniform(-2,2), random.uniform(0.5,2)])
            cam2world = bproc.camera.CameraUtility.look_at(location, look_at)
            # Check view coverage
            coverage = bproc.camera.CameraUtility.estimate_scene_coverage(cam2world, valid_objs)
            if coverage >= 0.4:
                cam_poses.append(cam2world)
                break

    for pose in cam_poses:
        bproc.camera.add_camera_pose(pose)

    # 9. Enable normal, depth, segmentation rendering, add alpha channel
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.enable_alpha_channel()

    # 10. Render and write to .hdf5
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```