```python
import numpy as np
from vispy import app, gloo
import time

vertex_shader = """
#version 120
attribute vec2 a_position;
attribute vec3 a_color;
attribute float a_size;
varying vec3 v_color;
void main()
{
    v_color = a_color;
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = a_size;
}
"""

fragment_shader = """
#version 120
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
        app.Canvas.__init__(self, keys='interactive', size=(800, 600), title='Fireworks Simulation')
        self.program = gloo.Program(vertex_shader, fragment_shader)
        self.n_particles = 0
        self.explosion_start = time.time()
        self.duration = 1.0  # seconds per explosion
        self.create_explosion()
        self.timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def create_explosion(self):
        # Random explosion center in NDC
        cx = np.random.uniform(-0.8, 0.8)
        cy = np.random.uniform(-0.8, 0.8)
        n = np.random.randint(80, 200)
        self.n_particles = n
        angles = np.random.uniform(0, 2 * np.pi, n)
        speeds = np.random.uniform(0.3, 1.0, n)
        colors = np.random.uniform(0.5, 1.0, (n, 3))
        sizes = np.random.uniform(4, 10, n)
        self.p0 = np.column_stack([np.full(n, cx), np.full(n, cy)])
        self.v = np.column_stack([np.cos(angles) * speeds, np.sin(angles) * speeds])
        self.colors = colors
        self.sizes = sizes
        self.explosion_start = time.time()

    def on_draw(self, event):
        gloo.clear('black')
        t = time.time() - self.explosion_start
        t = np.clip(t, 0, self.duration)
        # Gravity effect
        gravity = np.array([0, -0.8])
        positions = self.p0 + self.v * t + 0.5 * gravity * t**2
        fade = np.clip(1.0 - t / self.duration, 0, 1)
        # Update VBO
        self.program['a_position'] = positions.astype(np.float32)
        self.program['a_color'] = (self.colors * fade).astype(np.float32)
        self.program['a_size'] = (self.sizes * fade + 2).astype(np.float32)
        self.program.draw('points')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.size)

    def on_timer(self, event):
        t = time.time() - self.explosion_start
        if t > self.duration:
            self.create_explosion()
        self.update()

    def on_mouse_press(self, event):
        self.create_explosion()

if __name__ == '__main__':
    c = Canvas()
    app.run()
```