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

def sample_6dof_pose(bvh_tree, min_dist=0.3, max_trials=100):
    for _ in range(max_trials):
        location = np.random.uniform([-1, -1, 1], [1, 1, 2])
        rotation = bproc.math.build_transformation_mat(np.zeros(3), bproc.math.euler2quat(np.random.uniform(0, 2*np.pi, 3)))
        mat = bproc.math.build_transformation_mat(location, rotation)
        if bvh_tree.visible_from_camera(mat[:3, 3], min_dist):
            return mat
    return None

def main():
    args = parse_args()
    bop_parent_dir = args.bop_parent_dir
    cc_textures_dir = args.cc_textures_dir
    output_dir = args.output_dir
    num_scenes = args.num_scenes

    bproc.init()

    # Load BOP objects from 'itodd' and 'tless'
    itodd_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(bop_parent_dir, 'itodd'),
        model_type='eval',
        num_of_objs_to_sample=5
    )
    tless_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(bop_parent_dir, 'tless'),
        model_type='eval',
        num_of_objs_to_sample=5
    )
    bop_objs = itodd_objs + tless_objs

    # Load BOP intrinsics (using 'itodd' as reference)
    intrinsics = bproc.loader.load_bop_intrinsics(
        bop_dataset_path=os.path.join(bop_parent_dir, 'itodd')
    )

    # Set shading and hide objects
    for obj in bop_objs:
        obj.set_shading_mode('auto')
        obj.hide(True)

    # Create room using primitive planes
    room_planes = []
    size = 2.5
    thickness = 0.1
    # Floor
    floor = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    floor.set_location([0, 0, 0])
    room_planes.append(floor)
    # Ceiling
    ceiling = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    ceiling.set_location([0, 0, 2.5])
    ceiling.rotate_local(bproc.math.euler2quat([np.pi, 0, 0]))
    room_planes.append(ceiling)
    # Walls
    wall1 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall1.set_location([0, -size, 1.25])
    wall1.rotate_local(bproc.math.euler2quat([np.pi/2, 0, 0]))
    room_planes.append(wall1)
    wall2 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall2.set_location([0, size, 1.25])
    wall2.rotate_local(bproc.math.euler2quat([-np.pi/2, 0, 0]))
    room_planes.append(wall2)
    wall3 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall3.set_location([-size, 0, 1.25])
    wall3.rotate_local(bproc.math.euler2quat([0, np.pi/2, 0]))
    room_planes.append(wall3)
    wall4 = bproc.object.create_primitive('PLANE', scale=[size, size, 1])
    wall4.set_location([size, 0, 1.25])
    wall4.rotate_local(bproc.math.euler2quat([0, -np.pi/2, 0]))
    room_planes.append(wall4)

    # Enable rigidbody for planes
    for plane in room_planes:
        plane.enable_rigidbody(False, friction=0.8, linear_damping=0.8, angular_damping=0.8)

    # Create a light plane
    light_plane = bproc.object.create_primitive('PLANE', scale=[0.5, 0.5, 1])
    light_plane.set_location([0, 0, 2.4])
    light_plane_material = bproc.material.create('light_plane_mat')
    light_plane_material.make_emissive(emission_strength=10, emission_color=[1, 1, 1])
    light_plane.replace_materials(light_plane_material)

    # Create a point light
    point_light = bproc.types.Light()
    point_light.set_type("POINT")
    point_light.set_location([0, 0, 2.2])
    point_light.set_energy(100)

    # Load cc textures
    cc_textures = [os.path.join(cc_textures_dir, f) for f in os.listdir(cc_textures_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Enable depth rendering without antialiasing
    bproc.renderer.enable_depth_output(antialiasing=False)
    bproc.renderer.set_max_amount_of_samples(64)

    bop_writer = bproc.writer.BopWriter(
        output_dir=output_dir,
        dataset="itodd_tless",
        depth_scale=1000,
        append_to_existing_output=False
    )

    for scene_id in range(num_scenes):
        # Unhide and randomize bop objects
        for obj in bop_objs:
            obj.hide(False)
            bproc.material.random_sample_materials_for_each_obj([obj])

        # Sample a subset of objects for this scene
        scene_objs = random.sample(bop_objs, k=random.randint(3, 6))

        # Set physics for objects
        for obj in scene_objs:
            obj.enable_rigidbody(True, mass=1.0, friction=0.8, linear_damping=0.8, angular_damping=0.8)

        # Sample two light sources
        # Randomize light plane
        light_plane.set_location([random.uniform(-1, 1), random.uniform(-1, 1), 2.4])
        light_plane_material.make_emissive(emission_strength=random.uniform(8, 15), emission_color=np.random.uniform(0.8, 1.0, 3))
        # Randomize point light
        point_light.set_location([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(2.0, 2.4)])
        point_light.set_energy(random.uniform(80, 150))

        # Assign random cc texture to room planes
        for plane in room_planes:
            tex_path = random.choice(cc_textures)
            mat = bproc.material.create('room_plane_mat')
            mat.set_principled_shader_value("Base Color", tex_path)
            plane.replace_materials(mat)

        # Sample object poses
        bvh_room = bproc.object.create_bvh_tree_multi_objects(room_planes)
        poses = bproc.object.sample_poses_on_surface(
            objects_to_sample=scene_objs,
            surface_objects=room_planes,
            min_distance=0.05,
            max_distance=0.5,
            max_tries=100
        )
        for obj, pose in zip(scene_objs, poses):
            obj.set_local2world_mat(pose)

        # Check collisions
        bproc.object.perform_and_condition_check(scene_objs, room_planes)

        # Simulate physics and fix final poses
        bproc.object.simulate_physics_and_fix_final_poses(
            objects=scene_objs,
            min_simulation_time=2,
            max_simulation_time=4,
            check_object_interval=1
        )

        # Create BVH tree for camera obstacle checks
        bvh_all = bproc.object.create_bvh_tree_multi_objects(scene_objs + room_planes)

        # Generate camera poses
        cam_poses = []
        for _ in range(5):
            cam_pose = sample_6dof_pose(bvh_all, min_dist=0.3)
            if cam_pose is not None:
                cam_poses.append(cam_pose)

        # Set up camera intrinsics
        bproc.camera.set_intrinsics_from_blender_params(
            lens=intrinsics['cam_K'][0, 0],
            sensor_width=intrinsics['width'],
            sensor_height=intrinsics['height'],
            resolution_x=intrinsics['width'],
            resolution_y=intrinsics['height']
        )

        # Render pipeline
        data = bproc.renderer.render(
            render_pose=cam_poses,
            return_data=["colors", "depth", "segmentation", "normals"]
        )

        # Write data in bop format
        bop_writer.write_bop(
            scene_id=scene_id,
            images=data["colors"],
            depths=data["depth"],
            segmentations=data["segmentation"],
            normals=data["normals"],
            camera_poses=cam_poses,
            objects=scene_objs
        )

        # Disable rigidbody and hide objects
        for obj in bop_objs:
            obj.enable_rigidbody(False)
            obj.hide(True)

if __name__ == "__main__":
   