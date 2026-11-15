```python
import argparse
import os
import random
import numpy as np
import blenderproc as bproc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bop_parent_dir', type=str, required=True, help='Path to BOP datasets parent directory')
    parser.add_argument('--cc_textures_dir', type=str, required=True, help='Path to cc_textures directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num_scenes', type=int, required=True, help='Number of scenes to generate')
    return parser.parse_args()

def sample_6dof_pose(obj_extent, room_bounds):
    # Sample a random 6-DoF pose within the room bounds
    x = random.uniform(room_bounds['xmin'] + obj_extent, room_bounds['xmax'] - obj_extent)
    y = random.uniform(room_bounds['ymin'] + obj_extent, room_bounds['ymax'] - obj_extent)
    z = random.uniform(room_bounds['zmin'] + obj_extent, room_bounds['zmax'] - obj_extent)
    rotation = bproc.math.build_transformation_mat(
        np.zeros(3),
        bproc.math.euler2quat(np.random.uniform(0, 2*np.pi, 3))
    )[:3, :3]
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = [x, y, z]
    return pose

def main():
    args = parse_args()
    bproc.init()

    # Load BOP objects from 'itodd' and 'tless'
    bop_objs = []
    for dataset in ['itodd', 'tless']:
        bop_objs += bproc.loader.load_bop_objects(
            os.path.join(args.bop_parent_dir, dataset, 'models'),
            obj_ids=None
        )

    # Load intrinsics from BOP dataset
    intrinsics = bproc.loader.load_bop_intrinsics(
        os.path.join(args.bop_parent_dir, 'tless', 'camera.json')
    )

    # Set shading and hide objects
    for obj in bop_objs:
        obj.set_shading_mode('auto')
        obj.hide(True)

    # Create room using primitive planes
    room_planes = []
    room_size = 2.0
    thickness = 0.1
    # Floor
    floor = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    floor.set_location([0, 0, 0])
    room_planes.append(floor)
    # Ceiling
    ceiling = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    ceiling.set_location([0, 0, room_size])
    ceiling.rotate_local(bproc.math.euler2quat([np.pi, 0, 0]))
    room_planes.append(ceiling)
    # Walls
    wall1 = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    wall1.set_location([0, -room_size, room_size/2])
    wall1.rotate_local(bproc.math.euler2quat([np.pi/2, 0, 0]))
    room_planes.append(wall1)
    wall2 = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    wall2.set_location([0, room_size, room_size/2])
    wall2.rotate_local(bproc.math.euler2quat([-np.pi/2, 0, 0]))
    room_planes.append(wall2)
    wall3 = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    wall3.set_location([-room_size, 0, room_size/2])
    wall3.rotate_local(bproc.math.euler2quat([0, np.pi/2, 0]))
    room_planes.append(wall3)
    wall4 = bproc.object.create_primitive('PLANE', scale=[room_size, room_size, 1])
    wall4.set_location([room_size, 0, room_size/2])
    wall4.rotate_local(bproc.math.euler2quat([0, -np.pi/2, 0]))
    room_planes.append(wall4)

    # Enable rigidbody for planes
    for plane in room_planes:
        plane.enable_rigidbody(False, friction=0.8, restitution=0.5)

    # Create a light plane
    light_plane = bproc.object.create_primitive('PLANE', scale=[0.5, 0.5, 1])
    light_plane.set_location([0, 0, room_size - 0.01])
    light_plane_material = bproc.material.create('light_plane_mat')
    light_plane_material.make_emissive(emission_strength=10, emission_color=[1, 1, 1, 1])
    light_plane.replace_materials(light_plane_material)

    # Create a point light
    bproc.types.Light(
        light_type="POINT",
        location=[0, 0, room_size - 0.2],
        energy=100
    )

    # Load cc textures
    cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_dir)

    # Room bounds for pose sampling
    room_bounds = {
        'xmin': -room_size + 0.2, 'xmax': room_size - 0.2,
        'ymin': -room_size + 0.2, 'ymax': room_size - 0.2,
        'zmin': 0.1, 'zmax': room_size - 0.2
    }

    # Enable depth rendering without antialiasing
    bproc.renderer.enable_depth_output(antialiasing=False)
    # Set max samples for color rendering
    bproc.renderer.set_max_amount_of_samples(128)

    for scene_id in range(args.num_scenes):
        # Sample BOP objects
        num_objs = random.randint(3, 6)
        sampled_objs = random.sample(bop_objs, num_objs)
        for obj in sampled_objs:
            obj.hide(False)
            obj.set_shading_mode('auto')
            obj.set_location([0, 0, 2])  # Start above the room
            obj.set_scale([1, 1, 1])
            obj.enable_rigidbody(True, mass=1.0, friction=0.8, restitution=0.5)

        # Randomize materials
        for obj in sampled_objs:
            if len(obj.get_materials()) > 0:
                mat = random.choice(cc_textures)
                obj.replace_materials(mat)

        # Sample two light sources
        for _ in range(2):
            pos = [
                random.uniform(room_bounds['xmin'], room_bounds['xmax']),
                random.uniform(room_bounds['ymin'], room_bounds['ymax']),
                random.uniform(room_size - 0.2, room_size)
            ]
            bproc.types.Light(
                light_type="POINT",
                location=pos,
                energy=random.uniform(50, 200)
            )

        # Assign random cc texture to room planes
        for plane in room_planes:
            mat = random.choice(cc_textures)
            plane.replace_materials(mat)

        # Sample object poses, check collisions, simulate physics, fix final poses
        for obj in sampled_objs:
            extent = max(obj.get_bound_box()[1] - obj.get_bound_box()[0])
            pose = sample_6dof_pose(extent, room_bounds)
            obj.set_local2world_mat(pose)
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=2,
            max_simulation_time=4,
            check_object_interval=1
        )

        # Create BVH tree for camera obstacle checks
        bvh_tree = bproc.object.create_bvh_tree_multi_objects(room_planes + sampled_objs)

        # Generate camera poses ensuring obstacles are at least 0.3m away
        cam_poses = []
        num_cam_poses = 5
        for _ in range(num_cam_poses):
            for _ in range(100):
                cam_pos = [
                    random.uniform(room_bounds['xmin'], room_bounds['xmax']),
                    random.uniform(room_bounds['ymin'], room_bounds['ymax']),
                    random.uniform(room_size/2, room_size - 0.1)
                ]
                look_at = [0, 0, room_size/2]
                cam2world = bproc.camera.CameraUtility.build_transformation_mat(cam_pos, look_at)
                min_dist = bvh_tree.min_distance(cam_pos)
                if min_dist > 0.3:
                    cam_poses.append(cam2world)
                    break

        # Render pipeline
        bproc.camera.add_camera_pose(cam_poses)
        data = bproc.renderer.render()

        # Write data in BOP format
        bproc.writer.write_bop(
            os.path.join(args.output_dir, f'scene_{scene_id:06d}'),
            sampled_objs,
            data,
            m2mm=True,
            append_to_existing_output=False
        )

        # Disable rigidbody and hide objects after each scene
        for obj in sampled_objs:
            obj.disable_rigidbody()
            obj.hide(True)

if __name__ == "__main__":
    main()
```