```python
import argparse
import os
import random
import numpy as np
import blenderproc as bproc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bop_parent_dir', type=str, required=True, help='Path to BOP datasets parent directory')
    parser.add_argument('--cc_textures_dir', type=str, required=True, help='Path to cc textures directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_scenes', type=int, required=True, help='Number of scenes to generate')
    return parser.parse_args()

def sample_6dof_pose(bvh_tree, min_dist=0.3, max_tries=100):
    for _ in range(max_tries):
        location = np.random.uniform([-1, -1, 0.5], [1, 1, 1.5])
        rotation = bproc.math.build_transformation_mat(
            location, bproc.math.euler2mat(np.random.uniform(0, 2*np.pi, 3))
        )
        if bvh_tree is not None:
            min_dist_to_obstacle = bvh_tree.compute_point_distance(location)
            if min_dist_to_obstacle > min_dist:
                return rotation
        else:
            return rotation
    return None

def main():
    args = parse_args()
    bproc.init()

    # Load BOP objects from 'itodd' and 'tless'
    bop_objs_itodd = bproc.loader.load_bop_objs(
        os.path.join(args.bop_parent_dir, "itodd"), model_type="eval"
    )
    bop_objs_tless = bproc.loader.load_bop_objs(
        os.path.join(args.bop_parent_dir, "tless"), model_type="eval"
    )
    bop_objs = bop_objs_itodd + bop_objs_tless

    # Load BOP intrinsics (using 'itodd' as reference)
    intrinsics = bproc.loader.load_bop_intrinsics(
        os.path.join(args.bop_parent_dir, "itodd", "camera_intrinsics.json")
    )
    bproc.camera.set_intrinsics_from_blender_params(
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"], intrinsics["height"], intrinsics["width"]
    )

    # Set shading and hide objects
    for obj in bop_objs:
        obj.set_shading_mode("auto")
        obj.hide(True)

    # Create room using primitive planes
    room_planes = []
    size = 2.0
    thickness = 0.05
    # Floor
    floor = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    floor.set_location([0, 0, 0])
    floor.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(floor)
    # Ceiling
    ceiling = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    ceiling.set_location([0, 0, 2])
    ceiling.set_rotation_euler([np.pi, 0, 0])
    ceiling.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(ceiling)
    # Walls
    wall1 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall1.set_location([0, -size, 1])
    wall1.set_rotation_euler([np.pi/2, 0, 0])
    wall1.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(wall1)
    wall2 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall2.set_location([0, size, 1])
    wall2.set_rotation_euler([-np.pi/2, 0, 0])
    wall2.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(wall2)
    wall3 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall3.set_location([-size, 0, 1])
    wall3.set_rotation_euler([0, np.pi/2, 0])
    wall3.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(wall3)
    wall4 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall4.set_location([size, 0, 1])
    wall4.set_rotation_euler([0, -np.pi/2, 0])
    wall4.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(wall4)

    # Create a light plane
    light_plane = bproc.object.create_primitive('PLANE', scale=[0.5, 0.5, 1])
    light_plane.set_location([0, 0, 1.9])
    light_plane.set_rotation_euler([np.pi, 0, 0])
    light_plane_material = bproc.material.create('light_plane_mat')
    light_plane_material.make_emissive(emission_strength=10, emission_color=[1, 1, 1])
    light_plane.replace_materials(light_plane_material)

    # Create a point light
    point_light = bproc.types.Light()
    point_light.set_type("POINT")
    point_light.set_location([0, 0, 1.5])
    point_light.set_energy(100)

    # Load cc textures
    cc_textures = [os.path.join(args.cc_textures_dir, f) for f in os.listdir(args.cc_textures_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Enable depth rendering without antialiasing
    bproc.renderer.enable_depth_output(antialiasing=False)
    # Set max samples for color rendering
    bproc.renderer.set_max_samples(64)

    bop_writer = bproc.writer.BopWriter(
        os.path.join(args.output_dir, "bop_data"),
        dataset="custom",
        depth_scale=1000,
        append_to_existing_output=False
    )

    for scene_id in range(args.num_scenes):
        # Unhide and randomize bop objects
        for obj in bop_objs:
            obj.hide(False)
            bproc.material.random_sample_materials_for_each_obj(obj)

        # Set physics for bop objects
        for obj in bop_objs:
            obj.enable_rigidbody(active=True, collision_shape='CONVEX_HULL', mass=1.0)

        # Sample two light sources
        # Randomize light plane
        light_plane.set_location([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(1.5, 2)
        ])
        light_plane_material.make_emissive(
            emission_strength=random.uniform(5, 20),
            emission_color=np.random.uniform(0.8, 1.0, 3)
        )
        # Randomize point light
        point_light.set_location([
            random.uniform(-1, 1),
            random.uniform(-1, 1),
            random.uniform(1, 2)
        ])
        point_light.set_energy(random.uniform(50, 200))

        # Assign random cc texture to room planes
        for plane in room_planes:
            tex_path = random.choice(cc_textures)
            mat = bproc.material.create('room_plane_mat')
            mat.set_principled_shader_value("Base Color", tex_path)
            plane.replace_materials(mat)

        # Sample object poses
        for obj in bop_objs:
            pose = bproc.sampler.sample_pose_on_surface(
                surface=floor, min_distance=0.1, max_distance=1.0, min_height=0.01, max_height=0.05
            )
            obj.set_local2world_mat(pose)

        # Check collisions and simulate physics
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2, max_simulation_time=4, check_object_interval=1
        )

        # Create BVH tree for camera obstacle checks
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(bop_objs + room_planes)

        # Generate camera poses ensuring obstacles are at least 0.3m away
        cam_poses = []
        for _ in range(5):
            pose = sample_6dof_pose(bvh_tree, min_dist=0.3)
            if pose is not None:
                cam_poses.append(pose)
        bproc.camera.add_camera_pose_batch(cam_poses)

        # Render pipeline
        data = bproc.renderer.render()

        # Write data in bop format
        bop_writer.write_bop(scene_id, data, bop_objs)

        # Disable rigidbody and hide objects after each scene
        for obj in bop_objs:
            obj.disable_rigidbody()
            obj.hide(True)

if __name__ == "__main__":
    main()
```