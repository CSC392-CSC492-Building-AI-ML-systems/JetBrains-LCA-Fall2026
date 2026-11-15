```python
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager
from pytransform3d.transformations import (
    plot_transform, transform_from, adjoint_from_transform, transform_from_pq
)
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.screw_transformations import (
    screw_axis_from_screw_parameters, dual_quaternion_from_screw_parameters,
    transform_from_dual_quaternion
)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_screw(ax, origin, screw_axis, magnitude, color='r', label=None):
    """
    Plot a screw axis (force/torque) as an arrow and a helix.
    """
    w = screw_axis[:3]
    v = screw_axis[3:]
    w_norm = np.linalg.norm(w)
    if w_norm > 1e-6:
        # Helix parameters
        theta = magnitude
        n_turns = int(np.ceil(abs(theta) / (2 * np.pi)))
        t = np.linspace(0, theta, 100)
        # Helix along w, radius proportional to |v|/|w|
        axis_dir = w / w_norm
        pitch = np.dot(axis_dir, v) / w_norm
        radius = np.linalg.norm(np.cross(axis_dir, v)) / w_norm
        # Orthonormal basis
        if np.allclose(axis_dir, [1, 0, 0]):
            ortho = np.array([0, 1, 0])
        else:
            ortho = np.array([1, 0, 0])
        u = np.cross(axis_dir, ortho)
        u /= np.linalg.norm(u)
        v_ = np.cross(axis_dir, u)
        # Helix points
        helix = (origin.reshape(3, 1) +
                 radius * np.cos(t) * u.reshape(3, 1) +
                 radius * np.sin(t) * v_.reshape(3, 1) +
                 pitch * t * axis_dir.reshape(3, 1))
        ax.plot(helix[0], helix[1], helix[2], color=color, linewidth=2)
        # Arrow for direction
        arrow_start = helix[:, -1]
        arrow_dir = axis_dir * 0.1 * magnitude
        ax.quiver(
            arrow_start[0], arrow_start[1], arrow_start[2],
            arrow_dir[0], arrow_dir[1], arrow_dir[2],
            color=color, linewidth=2, arrow_length_ratio=0.2, label=label
        )
    else:
        # Pure translation (force only)
        arrow_dir = v * magnitude
        ax.quiver(
            origin[0], origin[1], origin[2],
            arrow_dir[0], arrow_dir[1], arrow_dir[2],
            color=color, linewidth=2, arrow_length_ratio=0.2, label=label
        )

def plot_sphere(ax, center, radius=0.05, color='b', alpha=0.5):
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
    tcp_frame = [f for f in tm._frames if "tcp" in f.lower() or "tool" in f.lower()]
    if tcp_frame:
        tcp_frame = tcp_frame[0]
    else:
        tcp_frame = list(tm._frames)[-1]  # fallback: last frame

    # Set joint angles (example: 6-DOF robot)
    joint_names = tm.get_joint_names()
    q = np.array([0.0, -np.pi/4, np.pi/2, 0.0, np.pi/4, 0.0])  # Example config
    joint_cfg = dict(zip(joint_names, q))
    tm.set_joint_positions(joint_cfg)

    # Get TCP pose in base frame
    T_base_tcp = tm.get_transform(base_frame, tcp_frame)
    tcp_pos = T_base_tcp[:3, 3]
    tcp_rot = T_base_tcp[:3, :3]

    # Spherical mass at TCP
    mass = 1.0  # kg
    g = np.array([0, 0, -9.81])  # gravity
    force_tcp = mass * g  # Force in TCP frame
    torque_tcp = np.zeros(3)  # Assume force acts at TCP, no torque

    # Wrench at TCP (6D: [force; torque])
    wrench_tcp = np.hstack((force_tcp, torque_tcp))

    # Visualize wrench as a screw axis
    # For a pure force, screw axis is [0,0,0, force direction]
    force_dir = force_tcp / (np.linalg.norm(force_tcp) + 1e-12)
    screw_axis_tcp = np.hstack((np.zeros(3), force_dir))
    magnitude_tcp = np.linalg.norm(force_tcp) * 0.05  # scale for visualization

    # Transform wrench to base frame using adjoint
    Ad = adjoint_from_transform(T_base_tcp)
    wrench_base = Ad @ wrench_tcp

    # Visualize wrench in base frame as a screw axis
    force_base = wrench_base[:3]
    torque_base = wrench_base[3:]
    if np.linalg.norm(force_base) > 1e-6:
        force_dir_base = force_base / np.linalg.norm(force_base)
    else:
        force_dir_base = np.zeros(3)
    screw_axis_base = np.hstack((np.zeros(3), force_dir_base))
    magnitude_base = np.linalg.norm(force_base) * 0.05

    # Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = make_3d_axis(ax_s=fig.add_subplot(111, projection="3d"), unit="m")
    ax.set_title("6-DOF Robot Arm Wrench Visualization")

    # Plot robot
    tm.plot_frames_in(ax, frames=[base_frame, tcp_frame], s=0.08)
    tm.plot_graph(ax=ax, show_visuals=True, show_frames=False)

    # Plot TCP
    plot_transform(ax, T_base_tcp, s=0.1)
    ax.scatter([tcp_pos[0]], [tcp_pos[1]], [tcp_pos[2]], color='k', s=50, label="TCP")

    # Plot spherical mass at TCP
    plot_sphere(ax, tcp_pos, radius=0.05, color='b', alpha=0.6)

    # Plot wrench at TCP (in TCP frame)
    plot_screw(ax, tcp_pos, tcp_rot @ screw_axis_tcp, magnitude_tcp, color='r', label="Wrench (TCP)")

    # Plot wrench in base frame (at TCP position)
    plot_screw(ax, tcp_pos, screw_axis_base, magnitude_base, color='g', label="Wrench (Base)")

    ax.legend()
    ax.set_xlim([tcp_pos[0] - 0.5, tcp_pos[0] + 0.5])
    ax.set_ylim([tcp_pos[1] - 0.5, tcp_pos[1] + 0.5])
    ax.set_zlim([tcp_pos[2] - 0.5, tcp_pos[2] + 0.5])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    plt.tight_layout()
    plt.savefig("robot_wrench_visualization.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
```