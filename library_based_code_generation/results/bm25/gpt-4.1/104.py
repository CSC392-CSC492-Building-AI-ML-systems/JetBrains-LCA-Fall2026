```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import (
    transform_from, plot_transform, adjoint_from_transform, transform, vector_to_point
)
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.screw_transformations import (
    screw_axis_from_screw_parameters, plot_screw, screw_matrix_from_screw_axis
)

def plot_screw_transformation(ax, origin, screw_axis, theta, length=0.2, color="r", label=None):
    """
    Plot the transformation about and along a screw axis.
    """
    # Screw axis: [wx, wy, wz, vx, vy, vz]
    # theta: rotation about the axis (rad)
    # The screw axis is assumed to be normalized
    plot_screw(
        ax=ax,
        screw_axis=screw_axis,
        theta=theta,
        length=length,
        origin=origin,
        color=color,
        label=label
    )

def wrench_to_screw_axis(force, torque):
    """
    Convert a wrench (force, torque) to a screw axis.
    """
    force = np.asarray(force)
    torque = np.asarray(torque)
    norm_f = np.linalg.norm(force)
    if norm_f > 1e-8:
        w = force / norm_f
        v = torque / norm_f
        return np.hstack((w, v)), norm_f
    else:
        norm_t = np.linalg.norm(torque)
        if norm_t > 1e-8:
            w = torque / norm_t
            v = np.zeros(3)
            return np.hstack((w, v)), norm_t
        else:
            return np.zeros(6), 0.0

def main():
    # Load robot from URDF
    urdf_path = "robot.urdf"  # Replace with your URDF file path
    tm = UrdfTransformManager()
    tm.load_urdf(urdf_path)
    base_frame = tm.get_root()
    tcp_frame = "tool0"  # Replace with your TCP frame name

    # Set joint angles
    joint_names = tm.get_joint_names()
    joint_angles = np.zeros(len(joint_names))
    joint_dict = dict(zip(joint_names, joint_angles))
    tm.set_joint_angles(joint_dict)

    # Get TCP pose in base frame
    T_base_tcp = tm.get_transform(tcp_frame, base_frame)

    # Spherical mass at TCP
    mass = 1.0  # kg
    g = 9.81  # m/s^2
    force_tcp = np.array([0, 0, -mass * g])  # Force in TCP frame (gravity)
    torque_tcp = np.zeros(3)  # No torque for point mass

    # Visualize wrench at TCP (as a screw)
    screw_axis_tcp, screw_magnitude_tcp = wrench_to_screw_axis(force_tcp, torque_tcp)

    # Transform wrench to base frame using adjoint
    Ad_T = adjoint_from_transform(T_base_tcp)
    wrench_tcp = np.hstack((force_tcp, torque_tcp))
    wrench_base = Ad_T @ wrench_tcp
    force_base = wrench_base[:3]
    torque_base = wrench_base[3:]
    screw_axis_base, screw_magnitude_base = wrench_to_screw_axis(force_base, torque_base)

    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = make_3d_axis(ax_s=fig.add_subplot(111, projection="3d"), n_ticks=5)
    ax.set_title("6-DOF Robot Arm with Wrench Visualization")

    # Plot robot
    tm.plot_frames_in("base", ax=ax, s=0.05)
    tm.plot_robot(ax=ax, show_visuals=True, show_collision=False, alpha=0.5)

    # Plot TCP frame
    plot_transform(ax=ax, A2=T_base_tcp, s=0.1, name="TCP")

    # Plot spherical mass at TCP
    tcp_pos = T_base_tcp[:3, 3]
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    r = 0.03
    x = r * np.cos(u) * np.sin(v) + tcp_pos[0]
    y = r * np.sin(u) * np.sin(v) + tcp_pos[1]
    z = r * np.cos(v) + tcp_pos[2]
    ax.plot_surface(x, y, z, color="b", alpha=0.6)

    # Plot wrench as screw at TCP (in TCP frame)
    plot_screw_transformation(
        ax=ax,
        origin=tcp_pos,
        screw_axis=screw_axis_tcp,
        theta=screw_magnitude_tcp * 0.05,  # scale for visualization
        length=0.15,
        color="r",
        label="Wrench (TCP)"
    )

    # Plot wrench as screw in base frame (at TCP position)
    plot_screw_transformation(
        ax=ax,
        origin=tcp_pos,
        screw_axis=screw_axis_base,
        theta=screw_magnitude_base * 0.05,
        length=0.15,
        color="g",
        label="Wrench (Base)"
    )

    ax.legend()
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.0])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plt.tight_layout()
    plt.savefig("robot_wrench_visualization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
```