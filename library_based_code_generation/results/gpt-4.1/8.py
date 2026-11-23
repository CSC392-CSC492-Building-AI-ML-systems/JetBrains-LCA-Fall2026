```python
import blenderproc as bproc
import argparse
import os
import random
import csv

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
    objs = bproc.loader.load_blend(args.scene_path)

    # Label objects
    label_mapping = load_label_mapping(args.label_csv)
    for obj in objs:
        if obj.get_name() in label_mapping:
            obj.set_cp("category_id", label_mapping[obj.get_name()])

    # Load materials
    material_files = [os.path.join(args.material_dir, f) for f in os.listdir(args.material_dir) if f.endswith('.blend')]
    loaded_materials = []
    for mat_file in material_files:
        loaded_materials.extend(bproc.material.load_blend(mat_file))

    # Randomly assign materials to 40% of objects
    mesh_objs = [obj for obj in objs if obj.type == "MESH"]
    num_to_assign = int(0.4 * len(mesh_objs))
    objs_to_assign = random.sample(mesh_objs, num_to_assign)
    assigned_materials = set()
    for obj in objs_to_assign:
        mat = random.choice(loaded_materials)
        obj.set_material(mat)
        assigned_materials.add(mat)

    # Load textures for assigned materials
    texture_files = [os.path.join(args.texture_dir, f) for f in os.listdir(args.texture_dir) if f.lower().endswith(('.jpg', '.png'))]
    for mat in assigned_materials:
        tex_file = random.choice(texture_files)
        bproc.texture.assign_texture(mat, tex_file)

    # Extract floors and ceilings from wall objects
    for obj in mesh_objs:
        if "wall" in obj.get_name().lower():
            bbox = obj.get_bound_box()
            min_z = min([v[2] for v in bbox])
            max_z = max([v[2] for v in bbox])
            # Create floor
            floor = bproc.object.create_primitive('PLANE', scale=[2,2,1])
            floor.set_location([obj.get_location()[0], obj.get_location()[1], min_z])
            floor.set_cp("category_id", 1)  # 1 for floor
            # Create ceiling
            ceiling = bproc.object.create_primitive('PLANE', scale=[2,2,1])
            ceiling.set_location([obj.get_location()[0], obj.get_location()[1], max_z])
            ceiling.set_cp("category_id", 2)  # 2 for ceiling

    # Make all lamp objects emit light
    for obj in objs:
        if obj.type == "LIGHT" or "lamp" in obj.get_name().lower():
            obj.set_light_type("POINT")
            obj.set_light_energy(1000)

    # Make all ceiling objects emit a bit of light
    for obj in bproc.object.get_all():
        if obj.get_cp("category_id") == 2:
            mat = bproc.material.create("CeilingEmission")
            mat.make_emissive(emission_strength=2, emission_color=[1,1,1,1])
            obj.set_material(mat)

    # Create BVH tree
    bvh_tree = bproc.object.create_bvh_tree_multi_objects([obj for obj in bproc.object.get_all() if obj.type == "MESH"])

    # Sample camera locations above the floor
    cam_poses = []
    tries = 0
    while len(cam_poses) < 10 and tries < 100:
        tries += 1
        loc = bproc.sampler.upper_region(bproc.object.get_all(), min_height=1.5, max_height=2.0)
        rot = bproc.camera.rotation_from_forward_vec([random.uniform(-1,1), random.uniform(-1,1), -1])
        cam2world = bproc.math.build_transformation_mat(loc, rot)
        # Check for obstacles
        if not bproc.camera.visible_objects(cam2world, bvh_tree, max_distance=10):
            continue
        cam_poses.append(cam2world)

    # Ensure scene coverage score is not too low
    if len(cam_poses) < 5:
        raise RuntimeError("Not enough valid camera poses for good scene coverage.")

    # Set camera poses
    bproc.camera.add_camera_pose(cam_poses)

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    bproc.writer.write_hdf5(os.path.join(args.output_dir, "output.hdf5"), data)

if __name__ == "__main__":
    main()
```