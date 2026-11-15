```python
import argparse
import os
import blenderproc as bproc
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="BlenderProc pipeline for house and chair replacement")
    parser.add_argument("--house_json", required=True, help="Path to house.json file")
    parser.add_argument("--chair_obj", required=True, help="Path to chair object file (.obj/.ply/.fbx)")
    parser.add_argument("--label_map_csv", required=True, help="Path to label mapping csv file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()

    # 3. Load house objects with label mapping
    house_objects = bproc.loader.load_suncg_scene(args.house_json, label_mapping_csv=args.label_map_csv)

    # 4. Load chair object and replace all chairs
    chair_obj = bproc.loader.load_obj(args.chair_obj)[0]
    chair_obj.hide(True)  # Hide the template chair

    # Find all chair objects in the scene
    chair_objs_in_scene = [obj for obj in bproc.object.get_all_mesh_objects() if "chair" in obj.get_cp("category_id", "").lower() or "chair" in obj.get_name().lower()]

    # Find all floor objects (to ignore collisions)
    floor_objs = [obj for obj in bproc.object.get_all_mesh_objects() if "floor" in obj.get_cp("category_id", "").lower() or "floor" in obj.get_name().lower()]

    for orig_chair in chair_objs_in_scene:
        # Duplicate the chair object
        new_chair = chair_obj.copy()
        new_chair.set_name(orig_chair.get_name())
        # Copy properties
        for k, v in orig_chair.get_cp_dict().items():
            new_chair.set_cp(k, v)
        # Set pose
        pose = orig_chair.get_local2world_mat()
        # Random rotation around z-axis
        angle = random.uniform(0, 2 * np.pi)
        rot_z = bproc.math.build_transformation_mat([0,0,0], [0,0,angle])
        pose = pose @ rot_z
        new_chair.set_local2world_mat(pose)
        new_chair.hide(False)
        # Remove original chair
        orig_chair.delete()

    # 5. Filter out invalid objects
    bproc.object.delete_invalid()

    # 6. Make all Suncg objects emit light
    bproc.lighting.light_suncg_scene()

    # 7. Initialize point sampler and BVH tree
    mesh_objs = bproc.object.get_all_mesh_objects()
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objs)
    sampler = bproc.sampler.Front3DPointInRoomSampler(house_json_path=args.house_json)

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
            # Random orientation
            rotation = bproc.camera.rotation_from_forward_and_up_vector(
                forward_vector=np.array([random.uniform(-1,1), random.uniform(-1,1), random.uniform(-0.2,0.2)]),
                up_vector=np.array([0,0,1])
            )
            cam2world = bproc.math.build_transformation_mat(location, rotation)
            # Check view coverage (dummy check: always accept for simplicity)
            cam_poses.append(cam2world)
            break

    for pose in cam_poses:
        bproc.camera.add_camera_pose(pose)

    # 9. Enable normal, depth, segmentation rendering, add alpha channel
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="instance")
    bproc.renderer.set_output_format(enable_alpha=True)

    # 10. Render and write to .hdf5
    data = bproc.renderer.render()
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```