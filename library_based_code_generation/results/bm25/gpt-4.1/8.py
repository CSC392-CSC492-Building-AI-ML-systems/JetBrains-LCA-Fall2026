```python
import blenderproc as bproc
import argparse
import os
import random
import csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Path to the scene file')
    parser.add_argument('--texture_dir', required=True, help='Directory containing textures')
    parser.add_argument('--material_dir', required=True, help='Directory containing materials')
    parser.add_argument('--label_csv', required=True, help='CSV file mapping object names to category IDs')
    parser.add_argument('--output_dir', required=True, help='Output directory')
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
    objs = bproc.loader.load_blend(args.scene)

    # Load label mapping
    label_mapping = load_label_mapping(args.label_csv)

    # Label objects
    for obj in objs:
        if obj.get_name() in label_mapping:
            obj.set_cp("category_id", label_mapping[obj.get_name()])

    # Load materials
    material_files = [os.path.join(args.material_dir, f) for f in os.listdir(args.material_dir) if f.endswith('.blend')]
    loaded_materials = []
    for mat_file in material_files:
        loaded_materials.extend(bproc.material.load_and_create(mat_file))

    # Randomly assign materials to 40% of objects
    mesh_objs = [obj for obj in objs if obj.type == 'MESH']
    num_to_assign = int(0.4 * len(mesh_objs))
    objs_to_assign = random.sample(mesh_objs, num_to_assign)
    assigned_materials = set()
    for obj in objs_to_assign:
        mat = random.choice(loaded_materials)
        obj.set_material(mat)
        assigned_materials.add(mat)

    # Load textures for assigned materials
    texture_files = [os.path.join(args.texture_dir, f) for f in os.listdir(args.texture_dir)]
    for mat in assigned_materials:
        tex_file = random.choice(texture_files)
        bproc.material.attach_texture(mat, tex_file)

    # Extract floors and ceilings from wall objects and assign category IDs
    for obj in mesh_objs:
        if "wall" in obj.get_name().lower():
            floor, ceiling = bproc.object.extract_floor_and_ceiling(obj)
            if floor:
                floor.set_cp("category_id", 1)  # Example: 1 for floor
            if ceiling:
                ceiling.set_cp("category_id", 2)  # Example: 2 for ceiling

    # Make all lamp objects emit light
    lamp_objs = [obj for obj in objs if "lamp" in obj.get_name().lower()]
    for lamp in lamp_objs:
        bproc.object.light_suncg_scene(lamp, energy=1000)

    # Make all ceiling objects emit a bit of light
    for obj in mesh_objs:
        if "ceiling" in obj.get_name().lower():
            bproc.object.light_suncg_scene(obj, energy=50)

    # Create BVH tree for all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objs)

    # Sample camera locations above the floor
    floor_objs = [obj for obj in mesh_objs if obj.get_cp("category_id") == 1]
    cam_poses = []
    tries = 0
    while len(cam_poses) < 5 and tries < 100:
        tries += 1
        floor_obj = random.choice(floor_objs)
        min_pt, max_pt = bproc.object.min_and_max_point(floor_obj)
        x = random.uniform(min_pt[0], max_pt[0])
        y = random.uniform(min_pt[1], max_pt[1])
        z = max_pt[2] + 1.5
        cam_pose = bproc.camera.CameraPose([x, y, z], [x, y, z-1])
        # Check for obstacles in front of camera
        if bproc.object.perform_and_condition_check(cam_pose, bvh_tree):
            cam_poses.append(cam_pose)
    # Ensure scene coverage score is not too low
    if bproc.object.scene_coverage_score(cam_poses, mesh_objs) < 0.5:
        print("Warning: Scene coverage score is low.")

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output()

    # Render scene
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```