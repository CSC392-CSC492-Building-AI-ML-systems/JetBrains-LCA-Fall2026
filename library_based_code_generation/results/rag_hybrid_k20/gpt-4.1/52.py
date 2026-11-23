```python
import numpy as np
from vispy import app, gloo

vertex_shader = """
uniform float u_time;
uniform vec2 u_center;
uniform float u_explosion_time;
attribute vec2 a_position;
attribute vec2 a_velocity;
attribute vec3 a_color;
attribute float a_lifetime;
varying vec3 v_color;
void main()
{
    float t = min(u_time - u_explosion_time, a_lifetime);
    t = max(t, 0.0);
    vec2 pos = a_position + a_velocity * t + vec2(0.0, -0.5) * t * t; // gravity
    gl_Position = vec4((pos + u_center) / vec2(400.0, 300.0) - 1.0, 0.0, 1.0);
    gl_PointSize = 3.0;
    v_color = a_color;
}
"""

fragment_shader = """
varying vec3 v_color;
void main()
{
    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
    if (dist > 0.5)
        discard;
    gl_FragColor = vec4(v_color, 1.0 - dist);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), title='Vispy Fireworks')
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.n_particles = 0
        self.max_particles = 400
        self.explosion_time = 0.0
        self.center = np.array([0.0, 0.0], dtype=np.float32)
        self.particle_data = np.zeros(self.max_particles, dtype=[
            ('a_position', np.float32, 2),
            ('a_velocity', np.float32, 2),
            ('a_color',    np.float32, 3),
            ('a_lifetime', np.float32, 1)
        ])
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.t0 = app.default_timer()
        self.u_time = 0.0
        self.create_explosion()
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one'))
        self.show()

    def create_explosion(self):
        n = np.random.randint(80, self.max_particles)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        speeds = np.random.uniform(120, 320, n)
        vx = np.cos(angles) * speeds
        vy = np.sin(angles) * speeds
        colors = np.random.uniform(0.5, 1.0, (n, 3))
        colors /= colors.max(axis=1, keepdims=True)
        lifetimes = np.random.uniform(0.7, 1.0, n)
        self.center = np.random.uniform([-300, -200], [300, 200])
        self.particle_data['a_position'][:n] = 0
        self.particle_data['a_velocity'][:n, 0] = vx
        self.particle_data['a_velocity'][:n, 1] = vy
        self.particle_data['a_color'][:n] = colors
        self.particle_data['a_lifetime'][:n, 0] = lifetimes
        self.n_particles = n
        self.explosion_time = self.u_time
        self.program['u_explosion_time'] = self.explosion_time
        self.program['u_center'] = self.center.astype(np.float32)
        self.program.bind(gloo.VertexBuffer(self.particle_data[:n]))

    def on_draw(self, event):
        gloo.clear()
        self.program['u_time'] = self.u_time
        self.program.draw('points')

    def on_resize(self, event):
        width, height = event.size
        gloo.set_viewport(0, 0, width, height)

    def on_timer(self, event):
        self.u_time = app.default_timer() - self.t0
        if self.u_time - self.explosion_time > 1.0:
            self.create_explosion()
        self.update()

    def on_mouse_press(self, event):
        self.create_explosion()

if __name__ == '__main__':
    c = Canvas()
    app.run()
```