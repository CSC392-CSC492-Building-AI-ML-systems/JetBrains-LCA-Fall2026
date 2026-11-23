```python
from manim import *

class FirstScene(Scene):
    def construct(self):
        # Display LaTeX title and equation
        title = Tex(r"Fourier Transform", font_size=60)
        eq = MathTex(r"\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} dx", font_size=40)
        title.to_edge(UP)
        eq.next_to(title, DOWN)
        self.play(Write(title), FadeIn(eq))
        self.wait(1)
        # Transform title and fade out equation
        new_title = Tex(r"Frequency Domain", font_size=60).to_edge(UP)
        self.play(Transform(title, new_title), FadeOut(eq))
        self.wait(1)
        # Create grid and title
        grid = NumberPlane(x_range=[-4, 4], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.5})
        grid_title = Tex("Original Grid", font_size=36).to_edge(DOWN)
        self.play(Create(grid), FadeIn(grid_title))
        self.wait(1)
        # Apply non-linear function to grid
        def nl_func(point):
            x, y, z = point
            return np.array([x, y + 0.5 * np.sin(x), z])
        self.play(grid.animate.apply_function(nl_func))
        self.wait(1)
        # Transform grid title
        new_grid_title = Tex("Non-linear Transformation", font_size=36).to_edge(DOWN)
        self.play(Transform(grid_title, new_grid_title))
        self.wait(1)

class SecondScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.wait(0.5)
        self.play(Transform(square, circle))
        self.wait(0.5)
        self.play(FadeOut(square))
        self.wait(0.5)

class ThirdScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(0.5)
        def pointwise_func(point):
            x, y, z = point
            return np.array([x, y + 0.5 * np.sin(2 * x), z])
        self.play(square.animate.apply_function(pointwise_func))
        self.wait(1)

class FourthScene(Scene):
    def construct(self):
        text = Text("Euler's Identity", font_size=48).to_edge(UP)
        eq = MathTex(r"e^{i\pi} + 1 = 0", font_size=60).next_to(text, DOWN)
        self.play(Write(text), FadeIn(eq))
        self.wait(1)

class FifthScene(Scene):
    def construct(self):
        square = Square()
        decimal = DecimalNumber(0, num_decimal_places=2, font_size=36)
        decimal.next_to(square, RIGHT)
        self.add(square, decimal)
        def updater(mob):
            mob.next_to(square, RIGHT)
            mob.set_value(square.get_center()[0])
        decimal.add_updater(updater)
        self.play(square.animate.shift(RIGHT * 3), run_time=2)
        self.wait(0.5)
        decimal.remove_updater(updater)

class SixthScene(Scene):
    def construct(self):
        shapes = VGroup(
            Square(),
            Circle().shift(LEFT * 2),
            Triangle().shift(RIGHT * 2)
        )
        pi = MathTex(r"\pi", font_size=144).move_to(UP * 2)
        self.play(*[FadeIn(mob, scale=0.5) for mob in shapes], FadeIn(pi, scale=2))
        self.wait(0.5)
        # Spiral in
        for i, mob in enumerate(shapes):
            mob.save_state()
            mob.shift(DOWN * 3)
            mob.set_opacity(0)
            self.play(
                mob.animate.restore().set_opacity(1).rotate(PI * 2),
                run_time=1
            )
        self.wait(0.5)
        self.play(FadeOut(shapes), FadeOut(pi))
        self.wait(0.5)

class SeventhScene(Scene):
    def construct(self):
        # Miter joint
        tri1 = Polygon([0, 0, 0], [2, 0, 0], [1, 2, 0], color=BLUE, stroke_width=10, joint_type="miter").shift(LEFT * 3)
        # Bevel joint
        tri2 = Polygon([0, 0, 0], [2, 0, 0], [1, 2, 0], color=GREEN, stroke_width=10, joint_type="bevel")
        # Round joint
        tri3 = Polygon([0, 0, 0], [2, 0, 0], [1, 2, 0], color=RED, stroke_width=10, joint_type="round").shift(RIGHT * 3)
        self.play(Create(tri1), Create(tri2), Create(tri3))
        self.wait(1)
```