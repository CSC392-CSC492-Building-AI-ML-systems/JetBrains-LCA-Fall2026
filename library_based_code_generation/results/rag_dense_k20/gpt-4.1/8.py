```python
import blenderproc as bproc
import argparse
import os
import csv
import random
import numpy as np

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

    # Load label mapping and assign category ids
    label_mapping = load_label_mapping(args.label_csv)
    for obj in objs:
        if obj.get_name() in label_mapping:
            obj.set_cp("category_id", label_mapping[obj.get_name()])

    # Load materials
    haven_material_loader = bproc.material.HavenMaterialLoader(args.material_dir)
    haven_materials = haven_material_loader.get_materials()

    # Randomly assign materials to 40% of objects
    mesh_objs = [obj for obj in objs if obj.type == 'MESH']
    num_to_assign = int(0.4 * len(mesh_objs))
    objs_to_assign = random.sample(mesh_objs, num_to_assign)
    assigned_materials = set()
    for obj in objs_to_assign:
        mat = random.choice(haven_materials)
        bproc.material.test_new_random_haven_material(obj, mat)
        assigned_materials.add(mat)

    # Load textures for assigned materials
    for mat in assigned_materials:
        tex_files = [os.path.join(args.texture_dir, f) for f in os.listdir(args.texture_dir) if f.endswith(('.jpg', '.png'))]
        if tex_files:
            tex_file = random.choice(tex_files)
            bproc.material.test_create_material_from_texture(mat, tex_file)

    # Extract floors and ceilings from wall objects and assign category ids
    for obj in mesh_objs:
        if "wall" in obj.get_name().lower():
            # Dummy extraction: in practice, use geometry analysis
            if "floor" in obj.get_name().lower():
                obj.set_cp("category_id", 1)  # Example: 1 for floor
            elif "ceiling" in obj.get_name().lower():
                obj.set_cp("category_id", 2)  # Example: 2 for ceiling

    # Make all lamp objects emit light
    lamp_objs = [obj for obj in objs if obj.type == 'LIGHT']
    for lamp in lamp_objs:
        lamp.set_energy(1000.0)

    # Make all ceiling objects emit a bit of light
    for obj in mesh_objs:
        if "ceiling" in obj.get_name().lower():
            mat = obj.get_materials()[0] if obj.get_materials() else None
            if mat:
                mat.make_emissive(emission_strength=2.0, emission_color=[1.0, 1.0, 1.0])

    # Create BVH tree for all mesh objects
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(mesh_objs)

    # Sample camera locations above the floor
    room_sampler = bproc.camera.Front3DPointInRoomSampler(mesh_objs)
    camera_poses = []
    for _ in range(10):
        pose = room_sampler.sample_pose_above_floor(min_height=1.0, max_height=2.0)
        if pose is not None:
            # Check for obstacles in front of camera
            if not bproc.camera.is_obstacle_in_view(pose, bvh_tree):
                camera_poses.append(pose)
    # Ensure scene coverage score is not too low
    if len(camera_poses) < 3:
        raise RuntimeError("Not enough valid camera poses for good scene coverage.")

    # Enable normal, depth, and segmentation rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output()
    bproc.renderer.enable_segmentation_output(map_by="category_id")

    # Render the scene
    bproc.camera.add_camera_pose(np.array(camera_poses))
    data = bproc.renderer.render()

    # Write data to .hdf5 file
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "output.hdf5")
    bproc.writer.write_hdf5(output_path, data)

if __name__ == "__main__":
    main()
```