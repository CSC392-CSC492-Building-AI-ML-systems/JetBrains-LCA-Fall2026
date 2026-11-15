```python
import blenderproc as bproc
import argparse
import os
import csv
import random
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True)
    parser.add_argument('--texture_dir', type=str, required=True)
    parser.add_argument('--material_dir', type=str, required=True)
    parser.add_argument('--label_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def load_label_mapping(csv_path):
    mapping = {}
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mapping[row['object_name']] = int(row['category_id'])
    return mapping

def main():
    args = parse_args()
    bproc.init()

    # Load scene
    objs = bproc.loader.create_mesh_objects_from_file(args.scene_path, split_by='object')

    # Load label mapping and assign category ids
    label_mapping = load_label_mapping(args.label_csv)
    for obj in objs:
        obj_name = obj.get_name()
        if obj_name in label_mapping:
            obj.set_cp("category_id", label_mapping[obj_name])
        else:
            obj.set_cp("category_id", 0)

    # Load materials
    material_files = [os.path.join(args.material_dir, f) for f in os.listdir(args.material_dir) if f.endswith('.blend')]
    materials = []
    for mat_file in material_files:
        mats = bproc.material.convert_to_materials(mat_file)
        materials.extend(mats)

    # Randomly assign materials to 40% of objects
    num_objs = len(objs)
    num_to_assign = int(0.4 * num_objs)
    objs_to_assign = random.sample(objs, num_to_assign)
    assigned_materials = set()
    for obj in objs_to_assign:
        mat = random.choice(materials)
        obj.set_material(mat)
        assigned_materials.add(mat)

    # Load textures for assigned materials
    texture_files = [os.path.join(args.texture_dir, f) for f in os.listdir(args.texture_dir) if f.endswith(('.jpg', '.png'))]
    for mat in assigned_materials:
        tex_file = random.choice(texture_files)
        bproc.material.test_create_material_from_texture(tex_file, mat)

    # Extract floors and ceilings from wall objects and assign category ids
    for obj in objs:
        if "wall" in obj.get_name().lower():
            floor, ceiling = bproc.object._assign_materials_to_floor_wall_ceiling(obj)
            if floor:
                floor.set_cp("category_id", 1)  # Floor category id
            if ceiling:
                ceiling.set_cp("category_id", 2)  # Ceiling category id

    # Make all lamp objects emit light
    for obj in objs:
        if "lamp" in obj.get_name().lower():
            bproc.lighting.light_suncg_scene(obj, strength=1000)

    # Make all ceiling objects emit a bit of light
    for obj in objs:
        if obj.get_cp("category_id") == 2:
            emission_mat = bproc.material.create_emission_material(strength=5)
            obj.set_material(emission_mat)

    # Create BVH tree for all mesh objects
    mesh_objs = bproc.object.get_all_blender_mesh_objects()
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objs)

    # Sample camera locations above the floor
    sampler = bproc.camera.Front3DPointInRoomSampler(bvh_tree, min_height_above_floor=1.0, max_height_above_floor=2.0)
    camera_poses = []
    coverage_score = 0
    while coverage_score < 0.7 and len(camera_poses) < 10:
        pose = sampler.sample_pose()
        if pose is not None:
            camera_poses.append(pose)
            coverage_score = sampler.calc_coverage_score(camera_poses)

    # Set camera poses
    for pose in camera_poses:
        bproc.camera.add_camera_pose(pose)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    output_path = os.path.join(args.output_dir, "output.hdf5")
    bproc.writer.write_hdf5(output_path, data)

if __name__ == "__main__":
    main()
```