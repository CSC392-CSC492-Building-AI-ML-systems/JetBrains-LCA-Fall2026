```python
import numpy as np
from vispy import app, gloo

vertex_shader = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;
attribute float a_lifetime;
uniform float u_time;
uniform vec2 u_center;
varying vec3 v_color;
varying float v_alpha;
void main()
{
    float t = u_time / a_lifetime;
    t = clamp(t, 0.0, 1.0);
    vec2 pos = u_center + a_position * t;
    gl_Position = vec4(2.0 * pos.x - 1.0, 1.0 - 2.0 * pos.y, 0.0, 1.0);
    gl_PointSize = a_size * (1.0 - t);
    v_color = a_color;
    v_alpha = 1.0 - t;
}
"""

fragment_shader = """
#version 120
varying vec3 v_color;
varying float v_alpha;
void main()
{
    float dist = length(gl_PointCoord - vec2(0.5, 0.5));
    if (dist > 0.5)
        discard;
    gl_FragColor = vec4(v_color, v_alpha * (1.0 - dist));
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), title='Vispy Fireworks')
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.n_particles = 0
        self.explosion_duration = 1.0
        self.explosion_start = 0.0
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.create_explosion()
        gloo.set_state(clear_color='black', blend=True, blend_func=('src_alpha', 'one'))
        self.show()

    def create_explosion(self):
        # Random explosion center
        margin = 0.15
        center = np.random.uniform([margin, margin], [1 - margin, 1 - margin])
        n = np.random.randint(80, 180)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        speeds = np.random.uniform(0.15, 0.45, n)
        positions = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speeds[:, None]
        colors = np.random.uniform(0.5, 1.0, (n, 3))
        colors *= np.random.uniform(0.7, 1.0, (n, 1))
        sizes = np.random.uniform(6, 16, n)
        lifetimes = np.full(n, self.explosion_duration)
        self.n_particles = n
        self.center = center
        self.positions = positions.astype(np.float32)
        self.colors = colors.astype(np.float32)
        self.sizes = sizes.astype(np.float32)
        self.lifetimes = lifetimes.astype(np.float32)
        self.explosion_start = app.default_timer()
        self.program['a_position'] = self.positions
        self.program['a_color'] = self.colors
        self.program['a_size'] = self.sizes
        self.program['a_lifetime'] = self.lifetimes
        self.program['u_center'] = self.center

    def on_draw(self, event):
        gloo.clear()
        now = app.default_timer()
        t = now - self.explosion_start
        if t > self.explosion_duration:
            self.create_explosion()
            t = 0.0
        self.program['u_time'] = t
        self.program.draw('points')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        self.update()

    def on_mouse_press(self, event):
        self.create_explosion()

if __name__ == '__main__':
    c = Canvas()
    app.run()
```