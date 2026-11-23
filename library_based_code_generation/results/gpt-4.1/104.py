```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import (
    transform_from, plot_transform, adjoint_from_transform, transform, vector_to_skew_symmetric
)
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.plot_screws import plot_screw
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_screw_axis(ax, origin, direction, moment, length=0.2, color='r', label=None):
    """Plot a screw axis (force/torque) as an arrow and a moment as a circle."""
    # Plot force vector
    ax.quiver(
        origin[0], origin[1], origin[2],
        direction[0], direction[1], direction[2],
        length=length, color=color, linewidth=2, arrow_length_ratio=0.2, label=label
    )
    # Plot torque as a circle (moment)
    if np.linalg.norm(moment) > 1e-6:
        # Circle in the plane perpendicular to moment
        theta = np.linspace(0, 2 * np.pi, 100)
        radius = length * 0.3
        moment_dir = moment / np.linalg.norm(moment)
        # Find two orthogonal vectors in the plane
        if np.allclose(moment_dir, [0, 0, 1]):
            v = np.array([1, 0, 0])
        else:
            v = np.cross(moment_dir, [0, 0, 1])
            v /= np.linalg.norm(v)
        w = np.cross(moment_dir, v)
        circle = origin[:, None] + radius * (np.outer(v, np.cos(theta)) + np.outer(w, np.sin(theta)))
        ax.plot(circle[0], circle[1], circle[2], color=color, linestyle='--')

def plot_sphere(ax, center, radius=0.05, color='b', alpha=0.6):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def plot_robot_and_wrench(
    urdf_path, joint_angles, wrench_tcp, tcp_link, base_link, mass_center_offset, save_path
):
    # Load robot model
    tm = UrdfTransformManager()
    with open(urdf_path, "r") as f:
        tm.load_urdf(f.read(), mesh_path=None)
    # Set joint angles
    joint_names = list(joint_angles.keys())
    joint_values = np.array([joint_angles[name] for name in joint_names])
    tm.set_joint_angles(joint_angles)

    # Get TCP pose in base frame
    T_base_tcp = tm.get_transform(base_link, tcp_link)
    tcp_pos = T_base_tcp[:3, 3]
    tcp_rot = T_base_tcp[:3, :3]

    # Spherical mass position in TCP frame
    mass_pos_tcp = mass_center_offset
    # Transform mass position to base frame
    mass_pos_base = transform(T_base_tcp, np.hstack((mass_pos_tcp, 1)))[:3]

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = make_3d_axis(ax_s=fig.add_subplot(111, projection='3d'), unit='m')
    ax.set_title("6-DOF Robot Arm with Wrench Visualization")

    # Plot robot
    tm.plot_frames_in("base", ax=ax, s=0.05)
    tm.plot_graph(ax=ax, show_visuals=True, show_collision=False, show_frames=False)

    # Plot TCP frame
    plot_transform(ax=ax, A2=T_base_tcp, s=0.08, name="TCP")

    # Plot spherical mass at TCP
    plot_sphere(ax, mass_pos_base, radius=0.05, color='b', alpha=0.7)

    # Plot wrench at TCP (in TCP frame)
    force_tcp = wrench_tcp[:3]
    torque_tcp = wrench_tcp[3:]
    plot_screw_axis(
        ax, tcp_pos, tcp_rot @ force_tcp, tcp_rot @ torque_tcp, length=0.15, color='r', label="Wrench@TCP"
    )

    # Transform wrench to base frame using adjoint
    Ad = adjoint_from_transform(T_base_tcp)
    wrench_base = Ad @ wrench_tcp
    force_base = wrench_base[:3]
    torque_base = wrench_base[3:]

    # Plot wrench in base frame (at TCP position)
    plot_screw_axis(
        ax, tcp_pos, force_base, torque_base, length=0.15, color='g', label="Wrench@Base"
    )

    # Legend
    ax.legend(['TCP Wrench (red)', 'Base Wrench (green)'])

    # Set limits
    all_points = np.array([tcp_pos, mass_pos_base])
    for i in range(3):
        ax.set_xlim([np.min(all_points[:, 0]) - 0.2, np.max(all_points[:, 0]) + 0.2])
        ax.set_ylim([np.min(all_points[:, 1]) - 0.2, np.max(all_points[:, 1]) + 0.2])
        ax.set_zlim([np.min(all_points[:, 2]) - 0.2, np.max(all_points[:, 2]) + 0.2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# Example usage
if __name__ == "__main__":
    # Path to your URDF file
    urdf_path = "your_robot.urdf"  # Replace with your URDF file path

    # Example joint angles (replace with your robot's joint names and values)
    joint_angles = {
        "joint_1": 0.0,
        "joint_2": -0.5,
        "joint_3": 0.5,
        "joint_4": 0.0,
        "joint_5": 1.0,
        "joint_6": 0.0
    }

    # Wrench at TCP: [Fx, Fy, Fz, Tx, Ty, Tz] (in TCP frame)
    # Assume a mass of 1kg at 0.1m below TCP, gravity = 9.81 m/s^2
    mass = 1.0
    g = 9.81
    mass_center_offset = np.array([0.0, 0.0, -0.1])  # 10cm below TCP
    force_tcp = np.array([0.0, 0.0, -mass * g])
    torque_tcp = np.cross(mass_center_offset, force_tcp)
    wrench_tcp = np.hstack((force_tcp, torque_tcp))

    # Names of TCP and base link in your URDF
    tcp_link = "tool0"  # Replace with your TCP link name
    base_link = "base_link"  # Replace with your base link name

    # Output image path
    save_path = "robot_wrench_visualization.png"

    plot_robot_and_wrench(
        urdf_path, joint_angles, wrench_tcp, tcp_link, base_link, mass_center_offset, save_path
    )
```