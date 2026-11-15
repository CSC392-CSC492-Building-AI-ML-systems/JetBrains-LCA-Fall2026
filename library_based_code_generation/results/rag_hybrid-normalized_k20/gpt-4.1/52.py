```python
import numpy as np
from vispy import app, gloo

VERT_SHADER = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;
attribute float a_lifetime;
uniform float u_time;
varying vec3 v_color;
varying float v_alpha;
void main()
{
    float t = u_time / a_lifetime;
    t = clamp(t, 0.0, 1.0);
    // Fade out at the end of lifetime
    v_alpha = 1.0 - t;
    v_color = a_color;
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = a_size * (1.0 - t) + 1.0;
}
"""

FRAG_SHADER = """
#version 120
varying vec3 v_color;
varying float v_alpha;
void main()
{
    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
    if (dist > 0.5)
        discard;
    gl_FragColor = vec4(v_color, v_alpha);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), title='Vispy Fireworks')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.n_particles = 0
        self.max_particles = 1000
        self.particle_data = np.zeros(self.max_particles, dtype=[
            ('a_position', np.float32, 2),
            ('a_color',    np.float32, 3),
            ('a_size',     np.float32, 1),
            ('a_lifetime', np.float32, 1),
        ])
        self.velocities = np.zeros((self.max_particles, 2), dtype=np.float32)
        self.birth_time = np.zeros(self.max_particles, dtype=np.float32)
        self.active = np.zeros(self.max_particles, dtype=bool)
        self.time = 0.0
        self.last_explosion = -1.0
        self.explosion_interval = 1.0
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.program['u_time'] = 0.0
        self.set_current()
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one'))
        self.show()

    def on_initialize(self, event):
        pass

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)
        self.width = width
        self.height = height

    def on_draw(self, event):
        gloo.clear()
        if self.n_particles > 0:
            self.program.draw('points', vertices=self.n_particles)

    def on_timer(self, event):
        dt = event.dt if event.dt is not None else 1/60.
        self.time += dt
        # Remove dead particles
        alive = (self.time - self.birth_time) < self.particle_data['a_lifetime']
        self.active[:] = alive
        self.n_particles = np.count_nonzero(alive)
        if self.n_particles > 0:
            idx = np.where(alive)[0]
            # Update positions
            self.particle_data['a_position'][idx] += self.velocities[idx] * dt
            # Gravity
            self.velocities[idx,1] -= 0.5 * dt
            # Update VBO
            self.program.set_vars(**{k: self.particle_data[k][idx] for k in self.particle_data.dtype.names})
            self.program['u_time'] = self.time - self.birth_time[idx]
        # New explosion
        if self.time - self.last_explosion > self.explosion_interval:
            self.create_explosion()
            self.last_explosion = self.time
        self.update()

    def create_explosion(self):
        n = np.random.randint(60, 120)
        if n > self.max_particles:
            n = self.max_particles
        # Find available slots
        available = np.where(~self.active)[0]
        if len(available) < n:
            n = len(available)
        if n == 0:
            return
        idx = available[:n]
        # Center of explosion
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.6, 0.6)
        # Unique color
        base_color = np.random.uniform(0.5, 1.0, 3)
        base_color /= np.max(base_color)
        # Random directions
        angles = np.random.uniform(0, 2*np.pi, n)
        speeds = np.random.uniform(0.3, 0.7, n)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        # Lifetime and size
        lifetime = np.random.uniform(0.8, 1.2, n)
        size = np.random.uniform(6, 12, n)
        # Assign
        self.particle_data['a_position'][idx,0] = cx
        self.particle_data['a_position'][idx,1] = cy
        self.particle_data['a_color'][idx] = base_color * np.random.uniform(0.7, 1.0, (n,3))
        self.particle_data['a_size'][idx] = size
        self.particle_data['a_lifetime'][idx] = lifetime
        self.velocities[idx,0] = vx
        self.velocities[idx,1] = vy
        self.birth_time[idx] = self.time
        self.active[idx] = True
        self.n_particles = np.count_nonzero(self.active)
        # Update VBO
        idx_active = np.where(self.active)[0]
        self.program.set_vars(**{k: self.particle_data[k][idx_active] for k in self.particle_data.dtype.names})
        self.program['u_time'] = self.time - self.birth_time[idx_active]

    def on_mouse_press(self, event):
        # Create explosion at mouse position
        pos = event.pos
        x = (pos[0] / self.size[0]) * 2 - 1
        y = 1 - (pos[1] / self.size[1]) * 2
        self.create_explosion_at(x, y)

    def create_explosion_at(self, cx, cy):
        n = np.random.randint(60, 120)
        if n > self.max_particles:
            n = self.max_particles
        available = np.where(~self.active)[0]
        if len(available) < n:
            n = len(available)
        if n == 0:
            return
        idx = available[:n]
        base_color = np.random.uniform(0.5, 1.0, 3)
        base_color /= np.max(base_color)
        angles = np.random.uniform(0, 2*np.pi, n)
        speeds = np.random.uniform(0.3, 0.7, n)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        lifetime = np.random.uniform(0.8, 1.2, n)
        size = np.random.uniform(6, 12, n)
        self.particle_data['a_position'][idx,0] = cx
        self.particle_data['a_position'][idx,1] = cy
        self.particle_data['a_color'][idx] = base_color * np.random.uniform(0.7, 1.0, (n,3))
        self.particle_data['a_size'][idx] = size
        self.particle_data['a_lifetime'][idx] = lifetime
        self.velocities[idx,0] = vx
        self.velocities[idx,1] = vy
        self.birth_time[idx] = self.time
        self.active[idx] = True
        self.n_particles = np.count_nonzero(self.active)
        idx_active = np.where(self.active)[0]
        self.program.set_vars(**{k: self.particle_data[k][idx_active] for k in self.particle_data.dtype.names})
        self.program['u_time'] = self.time - self.birth_time[idx_active]

if __name__ == '__main__':
    c = Canvas()
    app.run()
```