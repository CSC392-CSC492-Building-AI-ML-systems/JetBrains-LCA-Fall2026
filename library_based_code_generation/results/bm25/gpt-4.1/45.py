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

def sample_6dof_pose(room_planes, bvh_tree, min_dist=0.3):
    for _ in range(100):
        location = np.random.uniform([-1.5, -1.5, 1.0], [1.5, 1.5, 2.0])
        rotation = bproc.math.build_transformation_mat([0, 0, 0], bproc.math.euler2quat(np.random.uniform(0, 2*np.pi, 3)))
        pose = np.eye(4)
        pose[:3, :3] = rotation[:3, :3]
        pose[:3, 3] = location
        min_obstacle_dist = bvh_tree.compute_point_distance(location)
        if min_obstacle_dist > min_dist:
            return pose
    return None

def main():
    args = parse_args()
    bproc.init()

    # Load BOP objects from 'itodd' and 'tless'
    bop_itodd_dir = os.path.join(args.bop_parent_dir, 'itodd')
    bop_tless_dir = os.path.join(args.bop_parent_dir, 'tless')
    itodd_objs = bproc.loader.load_bop_objects(bop_itodd_dir, model_type='eval')
    tless_objs = bproc.loader.load_bop_objects(bop_tless_dir, model_type='eval')
    bop_objs = itodd_objs + tless_objs

    # Load BOP intrinsics (using tless as example)
    intrinsics = bproc.loader.load_bop_intrinsics(bop_tless_dir)

    # Set shading and hide objects
    for obj in bop_objs:
        obj.set_shading_mode('auto')
        obj.hide(True)

    # Create room using primitive planes
    room_planes = []
    size = 2.0
    thickness = 0.05
    # Floor
    floor = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    floor.set_location([0, 0, 0])
    room_planes.append(floor)
    # Ceiling
    ceiling = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    ceiling.set_location([0, 0, 2])
    ceiling.rotate_local(bproc.math.euler2quat([np.pi, 0, 0]))
    room_planes.append(ceiling)
    # Walls
    for i, (loc, rot) in enumerate([
        ([size, 0, 1], [0, np.pi/2, 0]),
        ([-size, 0, 1], [0, -np.pi/2, 0]),
        ([0, size, 1], [np.pi/2, 0, 0]),
        ([0, -size, 1], [-np.pi/2, 0, 0])
    ]):
        wall = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
        wall.set_location(loc)
        wall.rotate_local(bproc.math.euler2quat(rot))
        room_planes.append(wall)

    # Enable rigidbody for planes
    for plane in room_planes:
        plane.enable_rigidbody(False, friction=0.8, restitution=0.1)

    # Create a light plane
    light_plane = bproc.object.create_primitive('PLANE', scale=[0.5, 0.5, 1])
    light_plane.set_location([0, 0, 1.95])
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create('light_plane_mat')
    light_plane_material.make_emissive(emission_strength=10, emission_color=[1, 1, 1])
    light_plane.replace_materials(light_plane_material)

    # Create a point light
    point_light = bproc.types.Light()
    point_light.set_type("POINT")
    point_light.set_location([0, 0, 1.8])
    point_light.set_energy(100)

    # Load cc textures
    cc_textures = bproc.loader.load_cc_textures(args.cc_textures_dir)

    # Enable depth rendering without antialiasing
    bproc.renderer.enable_depth_output(antialiasing=False)
    # Set max samples for color rendering
    bproc.renderer.set_max_amount_of_samples(64)

    for scene_idx in range(args.num_scenes):
        # Sample bop objects
        num_objs = random.randint(3, 6)
        sampled_objs = random.sample(bop_objs, num_objs)
        for obj in sampled_objs:
            obj.hide(False)
            bproc.material.random_sample_materials_for_each_obj(obj)

        # Set physics for objects
        for obj in sampled_objs:
            obj.enable_rigidbody(True, mass=1.0, friction=0.8, restitution=0.1)

        # Sample two light sources
        # Randomize light plane emission
        light_plane_material.make_emissive(
            emission_strength=random.uniform(5, 20),
            emission_color=np.random.uniform(0.8, 1.0, 3).tolist()
        )
        # Randomize point light
        point_light.set_location(np.random.uniform([-1, -1, 1.5], [1, 1, 2]))
        point_light.set_energy(random.uniform(50, 200))

        # Assign random cc texture to room planes
        for plane in room_planes:
            tex = random.choice(cc_textures)
            plane.replace_materials(tex)

        # Sample object poses
        for obj in sampled_objs:
            pose = bproc.sampler.sample_poses_on_surface(
                [floor], min_distance=0.1, max_distance=1.5, rotation_sample='normal'
            )
            obj.set_local2world_mat(pose)

        # Check collisions
        if not bproc.object.perform_and_condition_check(sampled_objs, bproc.object.check_collisions):
            for obj in sampled_objs:
                obj.hide(True)
            continue

        # Simulate physics and fix final poses
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2, max_simulation_time=4, check_object_interval=0.1
        )

        # Create BVH tree for camera obstacle checks
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(room_planes + sampled_objs)

        # Generate camera poses ensuring obstacles are at least 0.3m away
        cam_poses = []
        for _ in range(5):
            pose = sample_6dof_pose(room_planes, bvh_tree, min_dist=0.3)
            if pose is not None:
                cam_poses.append(pose)
        if not cam_poses:
            for obj in sampled_objs:
                obj.hide(True)
            continue

        # Render pipeline
        data = bproc.renderer.render(
            poses=cam_poses,
            intrinsics=intrinsics,
            render_depth=True,
            render_normals=True,
            render_segmentation=True
        )

        # Write data in bop format
        bproc.writer.write_bop(
            os.path.join(args.output_dir, f'scene_{scene_idx:06d}'),
            sampled_objs,
            data,
            cam_poses,
            intrinsics
        )

        # Disable rigidbody and hide objects
        for obj in sampled_objs:
            obj.disable_rigidbody()
            obj.hide(True)

if __name__ == "__main__":
    main()
```