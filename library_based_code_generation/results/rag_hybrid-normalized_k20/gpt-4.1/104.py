```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import (
    transform_from, plot_transform, adjoint_from_transform
)
from pytransform3d.screw_transformations import (
    screw_axis_from_screw_parameters, screw_parameters_from_screw_axis,
    screw_matrix_from_screw_axis, screw_axis_from_screw_matrix
)
from pytransform3d.plot_utils import make_3d_axis
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_screw(ax, origin, screw_axis, length=0.2, color='r', label=None):
    """
    Plot a screw axis (force/torque) as an arrow and a helix.
    """
    w = screw_axis[:3]
    v = screw_axis[3:]
    w_norm = np.linalg.norm(w)
    if w_norm > 1e-6:
        # Helix for torque
        theta = np.linspace(0, 2 * np.pi, 100)
        r = length * 0.1
        axis = w / w_norm
        # Find a vector perpendicular to axis
        perp = np.cross(axis, [1, 0, 0])
        if np.linalg.norm(perp) < 1e-3:
            perp = np.cross(axis, [0, 1, 0])
        perp = perp / np.linalg.norm(perp)
        # Build rotation matrix
        R = np.eye(3)
        R[:, 0] = axis
        R[:, 1] = perp
        R[:, 2] = np.cross(axis, perp)
        helix = np.zeros((3, len(theta)))
        helix[0] = theta * length / (2 * np.pi)
        helix[1] = r * np.cos(theta)
        helix[2] = r * np.sin(theta)
        helix = R @ helix + origin.reshape(3, 1)
        ax.plot(helix[0], helix[1], helix[2], color=color, linewidth=2)
        # Arrow for force
        ax.quiver(
            origin[0], origin[1], origin[2],
            v[0], v[1], v[2],
            length=length, color=color, arrow_length_ratio=0.2, linewidth=2, label=label
        )
    else:
        # Only force, plot as arrow
        ax.quiver(
            origin[0], origin[1], origin[2],
            v[0], v[1], v[2],
            length=length, color=color, arrow_length_ratio=0.2, linewidth=2, label=label
        )

def plot_sphere(ax, center, radius=0.05, color='b', alpha=0.6):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def main():
    # Load robot from URDF
    urdf_path = "robot.urdf"  # Replace with your URDF file path
    tm = UrdfTransformManager()
    tm.load_urdf(urdf_path)
    base_frame = tm.get_root()
    tcp_frame = list(tm.get_frame_names())[-1]  # Assume last frame is TCP

    # Set joint angles
    joint_names = tm.get_joint_names()
    joint_angles = np.zeros(len(joint_names))
    # Example: set some nonzero angles
    if len(joint_angles) >= 6:
        joint_angles[:6] = [0.5, -0.3, 0.7, 0.2, -0.5, 0.4]
    joint_cfg = dict(zip(joint_names, joint_angles))
    tm.set_joint_positions(joint_cfg)

    # Get TCP pose in base frame
    T_base_tcp = tm.get_transform(tcp_frame, base_frame)
    tcp_pos = T_base_tcp[:3, 3]

    # Spherical mass at TCP
    mass = 1.0  # kg
    g = np.array([0, 0, -9.81])  # gravity
    force_tcp = mass * g  # Force in TCP frame
    torque_tcp = np.zeros(3)  # No torque from mass at TCP

    # Wrench at TCP (6D vector: [torque; force])
    wrench_tcp = np.hstack((torque_tcp, force_tcp))

    # Visualize
    fig = plt.figure(figsize=(10, 8))
    ax = make_3d_axis(ax_s=fig.add_subplot(111, projection='3d'), n_ticks=5)
    ax.set_title("Wrench Visualization on 6-DOF Robot Arm")

    # Plot robot
    tm.plot_frames_in(ax, frames=[base_frame, tcp_frame], s=0.08)
    tm.plot_graph(ax=ax, show_visuals=True, show_frames=False)

    # Plot TCP
    plot_transform(ax, T_base_tcp, s=0.08, name="TCP")

    # Plot spherical mass at TCP
    plot_sphere(ax, tcp_pos, radius=0.05, color='b', alpha=0.6)

    # Plot wrench at TCP (in TCP frame)
    plot_screw(ax, tcp_pos, np.hstack((torque_tcp, force_tcp)), length=0.15, color='r', label="Wrench@TCP")

    # Transform wrench to base frame using adjoint
    Ad = adjoint_from_transform(T_base_tcp)
    wrench_base = Ad @ wrench_tcp

    # Plot wrench at TCP position, but in base frame (force/torque in base frame)
    plot_screw(ax, tcp_pos, wrench_base, length=0.15, color='g', label="Wrench@Base")

    # Legend
    ax.legend(loc='upper left')

    # Set limits
    ax.set_xlim([tcp_pos[0] - 0.5, tcp_pos[0] + 0.5])
    ax.set_ylim([tcp_pos[1] - 0.5, tcp_pos[1] + 0.5])
    ax.set_zlim([tcp_pos[2] - 0.5, tcp_pos[2] + 0.5])

    # Save image
    plt.tight_layout()
    plt.savefig("robot_wrench_visualization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
```