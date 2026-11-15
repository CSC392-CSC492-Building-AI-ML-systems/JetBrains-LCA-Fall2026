```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import (
    transform_from, plot_transform, adjoint_from_transform
)
from pytransform3d.screw_transformations import (
    screw_axis_from_screw_parameters, screw_parameters_from_screw_axis,
    screw_matrix_from_screw_axis, exponential_coordinates_from_screw_axis
)
from pytransform3d.plot_utils import make_3d_axis
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_screw(ax, origin, direction, moment, length=0.2, color='r', label=None):
    """
    Plot a screw (force/torque) as an arrow and a helix.
    """
    # Plot force arrow
    force = direction / np.linalg.norm(direction) * length
    ax.quiver(
        origin[0], origin[1], origin[2],
        force[0], force[1], force[2],
        color=color, linewidth=2, arrow_length_ratio=0.2, label=label
    )
    # Plot torque as a helix around the force direction
    n_points = 100
    t = np.linspace(0, 2 * np.pi, n_points)
    radius = 0.03 * length
    axis = direction / np.linalg.norm(direction)
    # Find two orthogonal vectors to axis
    if np.allclose(axis, [0, 0, 1]):
        v = np.array([1, 0, 0])
    else:
        v = np.cross(axis, [0, 0, 1])
        v /= np.linalg.norm(v)
    w = np.cross(axis, v)
    helix = (
        origin.reshape(3, 1)
        + radius * (np.outer(v, np.cos(t)) + np.outer(w, np.sin(t)))
        + (length * t / (2 * np.pi)) * axis.reshape(3, 1)
    )
    ax.plot(helix[0], helix[1], helix[2], color=color, linewidth=1)

def plot_sphere(ax, center, radius=0.05, color='b', alpha=0.5):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def plot_tcp(ax, tcp_pose, size=0.05):
    plot_transform(ax=ax, A=tcp_pose, s=size)

def plot_robot(tm, ax, joint_angles):
    tm.plot_frames_in("base", ax=ax, s=0.04)
    tm.plot_links(ax=ax, joint_angles=joint_angles, show_visuals=True)

def main():
    # Load robot from URDF
    urdf_path = "robot.urdf"  # Replace with your URDF file path
    tm = UrdfTransformManager()
    tm.load_urdf(urdf_path)
    base_frame = tm.get_root()
    tcp_frame = tm.get_tip()
    # Set joint angles
    joint_names = tm.get_joint_names()
    joint_angles = {name: 0.0 for name in joint_names}
    # Example: set some nonzero joint angles
    for i, name in enumerate(joint_names):
        joint_angles[name] = 0.2 * (i + 1)
    # Get TCP pose in base frame
    tcp_pose = tm.get_transform(tcp_frame, base_frame, joint_angles=joint_angles)
    tcp_position = tcp_pose[:3, 3]
    tcp_rotation = tcp_pose[:3, :3]
    # Spherical mass at TCP
    mass = 1.0  # kg
    g = np.array([0, 0, -9.81])  # gravity
    force_tcp = mass * g  # Force in TCP frame
    # Assume no external torque at TCP except from gravity at offset
    offset = np.array([0, 0, 0])  # If mass is offset from TCP, set here
    torque_tcp = np.cross(offset, force_tcp)
    # Wrench in TCP frame
    wrench_tcp = np.hstack((force_tcp, torque_tcp))
    # Transform wrench to base frame using adjoint
    Ad = adjoint_from_transform(tcp_pose)
    wrench_base = Ad.T @ wrench_tcp
    force_base = wrench_base[:3]
    torque_base = wrench_base[3:]
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = make_3d_axis(ax_s=fig.add_subplot(111, projection='3d'))
    # Plot robot
    plot_robot(tm, ax, joint_angles)
    # Plot TCP
    plot_tcp(ax, tcp_pose, size=0.07)
    # Plot spherical mass at TCP
    plot_sphere(ax, tcp_position, radius=0.05, color='b', alpha=0.6)
    # Plot wrench at TCP (in TCP frame)
    plot_screw(
        ax,
        tcp_position,
        tcp_rotation @ force_tcp,
        tcp_rotation @ torque_tcp,
        length=0.15,
        color='r',
        label='Wrench (TCP frame)'
    )
    # Plot wrench in base frame (at TCP position)
    plot_screw(
        ax,
        tcp_position,
        force_base,
        torque_base,
        length=0.15,
        color='g',
        label='Wrench (base frame)'
    )
    # Legend and view
    ax.legend()
    ax.set_title("6-DOF Robot Arm with Wrench Visualization")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.0])
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.savefig("robot_wrench_visualization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
```