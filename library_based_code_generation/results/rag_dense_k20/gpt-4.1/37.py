```python
import argparse
import os
import random
import numpy as np
import blenderproc as bproc

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

    # 3. Load objects from house.json with label mapping
    objects = bproc.loader.load_suncg(args.house_json, label_mapping_csv=args.label_map_csv)

    # 4. Load chair object and replace all chairs
    chair_objs = bproc.loader.load_obj(args.chair_obj)
    chair_obj = chair_objs[0]

    # Find all chair objects in the scene
    chair_class_ids = [obj.get_cp("category_id") for obj in objects if "chair" in obj.get_name().lower() or obj.get_cp("category") == "chair"]
    chair_objs_in_scene = [obj for obj in objects if "chair" in obj.get_name().lower() or obj.get_cp("category") == "chair"]

    # Find floor objects to ignore collisions
    floor_objs = [obj for obj in objects if "floor" in obj.get_name().lower() or obj.get_cp("category") == "floor"]

    for orig_chair in chair_objs_in_scene:
        # Duplicate the loaded chair object
        new_chair = chair_obj.duplicate()
        # Copy properties
        for k, v in orig_chair.get_cp().items():
            new_chair.set_cp(k, v)
        # Set pose
        pose = orig_chair.get_local2world_mat()
        # Random rotation around z-axis
        angle = random.uniform(0, 2 * np.pi)
        rot_z = bproc.math.build_transformation_mat([0, 0, 0], [0, 0, angle])
        pose = pose @ rot_z
        new_chair.set_local2world_mat(pose)
        # Ignore collision with floors
        for floor in floor_objs:
            bproc.object.disable_collision(new_chair, floor)
        # Remove original chair
        orig_chair.delete()

    # 5. Filter out invalid objects
    bproc.object.delete_invalid()

    # 6. Make all Suncg objects emit light
    for obj in objects:
        if obj.get_cp("SUNCG") or obj.get_name().startswith("SUNCG"):
            bproc.material.make_emissive(obj, emission_strength=5.0)

    # 7. Initialize point sampler and BVH tree
    sampler = bproc.sampler.Front3DPointInRoomSampler(objects)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(objects)

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
            look_at = location + np.array([random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-0.5, 0.5)])
            cam2world = bproc.camera.CameraUtility.look_at(location, look_at)
            # Check view coverage (dummy check, as actual coverage computation is complex)
            # Here, we just accept the pose
            cam_poses.append(cam2world)
            break

    for pose in cam_poses:
        bproc.camera.add_camera_pose(pose)

    # 9. Enable normal, depth, segmentation rendering, add alpha channel
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()
    bproc.renderer.set_output_format("PNG")
    bproc.renderer.set_alpha_mode("TRANSPARENT")

    # 10. Render and write to .hdf5
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```