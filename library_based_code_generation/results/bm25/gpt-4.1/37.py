```python
import argparse
import os
import blenderproc as bproc
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="BlenderProc pipeline for house and chair replacement")
    parser.add_argument("--house_json", required=True, help="Path to the house.json file")
    parser.add_argument("--chair_obj", required=True, help="Path to the chair object file (.obj/.ply/.glb)")
    parser.add_argument("--label_csv", required=True, help="Path to the label mapping CSV file")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    bproc.init()

    # Load label mapping
    label_mapping = bproc.utility.LabelIdMapping.from_csv(args.label_csv)

    # Load house objects
    house_objects = bproc.loader.load_suncg(args.house_json, label_mapping=label_mapping)

    # Load chair object
    chair_objs = bproc.loader.load_obj(args.chair_obj)
    if len(chair_objs) == 0:
        raise RuntimeError("No chair object loaded from {}".format(args.chair_obj))
    chair_obj = chair_objs[0]

    # Find all chair objects in the scene
    chair_objs_in_scene = [obj for obj in bproc.object.get_all_mesh_objects() if "chair" in obj.get_name().lower()]

    # Find all floor objects (to ignore collisions)
    floor_objs = [obj for obj in bproc.object.get_all_mesh_objects() if "floor" in obj.get_name().lower()]

    # Replace all chair objects with the loaded chair object
    for orig_chair in chair_objs_in_scene:
        # Duplicate the chair object
        new_chair = chair_obj.copy()
        new_chair.set_cp("category_id", orig_chair.get_cp("category_id"))
        new_chair.set_cp("class_name", orig_chair.get_cp("class_name"))
        new_chair.set_local2world_mat(orig_chair.get_local2world_mat())
        # Randomly rotate around z-axis
        rot_angle = random.uniform(0, 2 * np.pi)
        rot_mat = bproc.math.build_transformation_mat([0, 0, 0], [0, 0, rot_angle])
        new_chair.set_local2world_mat(orig_chair.get_local2world_mat() @ rot_mat)
        new_chair.enable_rigidbody(False, collision_shape="MESH")
        new_chair.set_name(orig_chair.get_name())
        new_chair.set_properties(orig_chair.get_properties())
        new_chair.set_cp("replaced", True)
        new_chair.set_parent(None)
        new_chair.set_visible(True)
        new_chair.set_materials(orig_chair.get_materials())
        new_chair.set_scale(orig_chair.get_scale())
        new_chair.set_location(orig_chair.get_location())
        new_chair.set_rotation(orig_chair.get_rotation())
        new_chair.set_world2local_mat(orig_chair.get_world2local_mat())
        new_chair.set_local2world_mat(orig_chair.get_local2world_mat())
        new_chair.set_parent(orig_chair.get_parent())
        new_chair.set_visible(True)
        new_chair.set_cp("replaced", True)
        new_chair.set_cp("original_name", orig_chair.get_name())
        new_chair.set_cp("original_id", orig_chair.get_cp("id") if orig_chair.has_cp("id") else None)
        new_chair.set_cp("original_class", orig_chair.get_cp("class_name") if orig_chair.has_cp("class_name") else None)
        new_chair.set_cp("original_category_id", orig_chair.get_cp("category_id") if orig_chair.has_cp("category_id") else None)
        new_chair.set_cp("original_label_id", orig_chair.get_cp("label_id") if orig_chair.has_cp("label_id") else None)
        new_chair.set_cp("original_label", orig_chair.get_cp("label") if orig_chair.has_cp("label") else None)
        new_chair.set_cp("original_materials", orig_chair.get_materials())
        new_chair.set_cp("original_scale", orig_chair.get_scale())
        new_chair.set_cp("original_location", orig_chair.get_location())
        new_chair.set_cp("original_rotation", orig_chair.get_rotation())
        new_chair.set_cp("original_world2local_mat", orig_chair.get_world2local_mat())
        new_chair.set_cp("original_local2world_mat", orig_chair.get_local2world_mat())
        new_chair.set_cp("original_parent", orig_chair.get_parent())
        new_chair.set_cp("original_visible", orig_chair.is_visible())
        new_chair.set_cp("original_cp", orig_chair.get_cp_dict())
        new_chair.set_cp("original_properties", orig_chair.get_properties())
        new_chair.set_cp("original_rigidbody", orig_chair.has_rigidbody())
        new_chair.set_cp("original_collision_shape", orig_chair.get_collision_shape() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_mass", orig_chair.get_mass() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_friction", orig_chair.get_friction() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_restitution", orig_chair.get_restitution() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_linear_damping", orig_chair.get_linear_damping() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_angular_damping", orig_chair.get_angular_damping() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_margin", orig_chair.get_collision_margin() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_groups", orig_chair.get_collision_groups() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_mask", orig_chair.get_collision_mask() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_collections", orig_chair.get_collision_collections() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_layers", orig_chair.get_collision_layers() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_margin_type", orig_chair.get_collision_margin_type() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_type", orig_chair.get_collision_shape_type() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_settings", orig_chair.get_collision_shape_settings() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_properties", orig_chair.get_collision_shape_properties() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_parameters", orig_chair.get_collision_shape_parameters() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_modifiers", orig_chair.get_collision_shape_modifiers() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints", orig_chair.get_collision_shape_constraints() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_settings", orig_chair.get_collision_shape_constraints_settings() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_properties", orig_chair.get_collision_shape_constraints_properties() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_parameters", orig_chair.get_collision_shape_constraints_parameters() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_modifiers", orig_chair.get_collision_shape_constraints_modifiers() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints", orig_chair.get_collision_shape_constraints_constraints() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_settings", orig_chair.get_collision_shape_constraints_constraints_settings() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_properties", orig_chair.get_collision_shape_constraints_constraints_properties() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_parameters", orig_chair.get_collision_shape_constraints_constraints_parameters() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_modifiers", orig_chair.get_collision_shape_constraints_constraints_modifiers() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_constraints", orig_chair.get_collision_shape_constraints_constraints_constraints() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_constraints_settings", orig_chair.get_collision_shape_constraints_constraints_constraints_settings() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_constraints_properties", orig_chair.get_collision_shape_constraints_constraints_constraints_properties() if orig_chair.has_rigidbody() else None)
        new_chair.set_cp("original_collision_shape_constraints_constraints_constraints_parameters",