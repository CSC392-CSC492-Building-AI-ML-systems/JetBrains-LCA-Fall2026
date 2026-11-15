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
        rotation = bproc.math.build_transformation_mat(np.zeros(3), bproc.math.euler2quat(np.random.uniform(0, 2*np.pi, 3)))
        if bvh_tree is not None:
            min_dist_to_obstacle = bvh_tree.compute_point_distance(location)
            if min_dist_to_obstacle > min_dist:
                return location, rotation
        else:
            return location, rotation
    raise RuntimeError("Could not sample a valid 6-DoF pose.")

def main():
    args = parse_args()
    bproc.init()

    bop_datasets = ['itodd', 'tless']
    bop_objs = []
    for dataset in bop_datasets:
        bop_objs += bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(args.bop_parent_dir, dataset),
            model_type='eval',
            num_of_objs_to_sample=5,
            split='train'
        )

    bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(args.bop_parent_dir, bop_datasets[0]),
        split='train'
    )

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
    floor.enable_rigidbody(active=False, collision_shape='BOX')
    room_planes.append(floor)
    # Ceiling
    ceiling = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    ceiling.set_location([0, 0, 2])
    ceiling.set_rotation_euler([np.pi, 0, 0])
    ceiling.enable_rigidbody(active=False, collision_shape='BOX')
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
        wall.set_rotation_euler(rot)
        wall.enable_rigidbody(active=False, collision_shape='BOX')
        room_planes.append(wall)

    # Light plane
    light_plane = bproc.object.create_primitive('PLANE', scale=[0.5, 0.5, 1])
    light_plane.set_location([0, 0, 1.95])
    light_plane.set_rotation_euler([0, 0, 0])
    light_plane_material = bproc.material.create('light_plane_mat')
    light_plane_material.make_emissive(emission_strength=10, emission_color=[1, 1, 1])
    light_plane.replace_materials(light_plane_material)

    # Point light
    bproc.types.LightSpot(location=[0, 0, 1.8], energy=1000)

    # Load cc textures
    cc_textures = bproc.loader.load_cc_textures(args.cc_textures_dir)

    # Enable depth rendering without antialiasing
    bproc.renderer.enable_depth_output(antialiasing=False)
    bproc.renderer.set_max_samples(64)

    bop_writer = bproc.writer.BopWriterUtility(
        output_dir=args.output_dir,
        dataset_name='custom_bop',
        depth_scale=1000,
        mask_encoding='polygon'
    )

    for scene_id in range(args.num_scenes):
        # Sample BOP objects
        sampled_objs = random.sample(bop_objs, k=min(5, len(bop_objs)))
        for obj in sampled_objs:
            obj.hide(False)
            bproc.material.random_sample_materials_for_each_obj(obj)

        # Set physics
        for obj in sampled_objs:
            obj.enable_rigidbody(active=True, collision_shape='CONVEX_HULL', mass=1.0)

        # Sample two light sources
        bproc.types.LightPoint(location=np.random.uniform([-1, -1, 1.5], [1, 1, 2.0]), energy=random.uniform(500, 1500))
        bproc.types.LightPoint(location=np.random.uniform([-1, -1, 1.5], [1, 1, 2.0]), energy=random.uniform(500, 1500))

        # Assign random cc texture to room planes
        texture = random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(texture)

        # Sample object poses
        bvh_tree_objs = bproc.object.create_bvh_tree_multi_objects(room_planes)
        for obj in sampled_objs:
            location, rotation = sample_6dof_pose(bvh_tree_objs)
            obj.set_location(location)
            obj.set_rotation_matrix(rotation[:3, :3])

        # Check collisions and simulate physics
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2,
            max_simulation_time=4,
            check_object_interval=1,
            settle_object_interval=1
        )

        # Create BVH tree for camera obstacle checks
        bvh_tree_all = bproc.object.create_bvh_tree_multi_objects(room_planes + sampled_objs)

        # Generate camera poses
        cam_poses = []
        num_cams = 5
        for _ in range(num_cams):
            for _ in range(100):
                cam_location = np.random.uniform([-1.5, -1.5, 1.0], [1.5, 1.5, 2.0])
                look_at = np.array([0, 0, 1])
                direction = look_at - cam_location
                direction = direction / np.linalg.norm(direction)
                if bvh_tree_all.compute_point_distance(cam_location) > 0.3:
                    cam2world = bproc.camera.CameraUtility.look_at(cam_location, look_at)
                    cam_poses.append(cam2world)
                    break

        # Render pipeline
        data = bproc.renderer.render()

        # Write data in bop format
        bop_writer.write_bop(scene_id, cam_poses, sampled_objs, data)

        # Disable rigidbody and hide objects
        for obj in sampled_objs:
            obj.enable_rigidbody(False)
            obj.hide(True)

if __name__ == "__main__":
    main()
```