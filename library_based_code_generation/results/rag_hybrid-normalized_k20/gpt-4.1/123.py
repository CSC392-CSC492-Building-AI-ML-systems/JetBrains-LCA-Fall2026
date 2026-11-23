```python
from manim import *

class TitleAndEquationScene(Scene):
    def construct(self):
        title = Tex(r"Fourier Transform", font_size=60)
        equation = MathTex(r"\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} dx", font_size=48)
        equation.next_to(title, DOWN, buff=0.75)
        self.play(Write(title), FadeIn(equation))
        self.wait(1)
        new_title = Tex(r"Frequency Domain", font_size=60).move_to(title)
        self.play(Transform(title, new_title), FadeOut(equation))
        self.wait(1)

        grid = NumberPlane(x_range=[-4, 4], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.5})
        grid_title = Tex("Original Grid", font_size=36).to_edge(UP)
        self.play(Create(grid), FadeIn(grid_title))
        self.wait(1)

        def nonlinear_func(point):
            x, y, z = point
            return np.array([x, y + 0.5 * np.sin(x), z])

        self.play(
            grid.animate.apply_function(nonlinear_func),
            run_time=2
        )
        new_grid_title = Tex("Nonlinear Transformation", font_size=36).to_edge(UP)
        self.play(Transform(grid_title, new_grid_title))
        self.wait(1)

class SquareToCircleScene(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.wait(0.5)
        self.play(Transform(square, circle))
        self.wait(0.5)
        self.play(FadeOut(square))
        self.wait(0.5)

class SquarePointwiseFunctionScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(0.5)

        def pointwise_func(point):
            x, y, z = point
            return np.array([x, y + 0.3 * np.sin(2 * x), z])

        self.play(square.animate.apply_function(pointwise_func), run_time=2)
        self.wait(1)

class TextAndEquationScene(Scene):
    def construct(self):
        text = Text("Euler's Identity", font_size=48)
        equation = MathTex(r"e^{i\pi} + 1 = 0", font_size=60)
        equation.next_to(text, DOWN, buff=0.75)
        self.play(Write(text), FadeIn(equation))
        self.wait(2)

class SquareAndDecimalUpdaterScene(Scene):
    def construct(self):
        square = Square().set_color(BLUE).move_to(LEFT * 3)
        decimal = DecimalNumber(0, num_decimal_places=2, font_size=48)
        decimal.next_to(square, UP)

        def updater(mob):
            mob.next_to(square, UP)
            mob.set_value(square.get_center()[0])

        decimal.add_updater(updater)
        self.add(square, decimal)
        self.play(square.animate.move_to(RIGHT * 3), run_time=3)
        self.wait(1)
        decimal.remove_updater(updater)

class SpiralShapesScene(Scene):
    def construct(self):
        shapes = VGroup(
            Square().set_color(RED),
            Circle().set_color(GREEN),
            Triangle().set_color(BLUE),
            Star(5).set_color(YELLOW)
        ).arrange(RIGHT, buff=1)
        pi = MathTex(r"\pi", font_size=144).set_color(PURPLE).move_to(ORIGIN)
        self.add(pi)
        for i, shape in enumerate(shapes):
            shape.scale(0.7)
            shape.move_to(3 * RIGHT * np.cos(i * PI / 2) + 2 * UP * np.sin(i * PI / 2))
            shape.set_opacity(0)
            self.add(shape)
        self.wait(0.5)

        anims = []
        for i, shape in enumerate(shapes):
            target = ORIGIN + 1.5 * RIGHT * np.cos(i * PI / 2) + 1 * UP * np.sin(i * PI / 2)
            anims.append(
                AnimationGroup(
                    shape.animate.set_opacity(1).move_to(target).rotate(PI * 2, about_point=ORIGIN),
                    lag_ratio=0.2
                )
            )
        self.play(*anims, FadeIn(pi, scale=2), run_time=2)
        self.wait(1)
        self.play(FadeOut(VGroup(*shapes, pi)), run_time=1)

class TrianglesWithLineJointsScene(Scene):
    def construct(self):
        tri1 = Polygon([0, 0, 0], [2, 0, 0], [1, 2, 0], color=WHITE, stroke_width=10, joint_type="miter")
        tri2 = Polygon([3, 0, 0], [5, 0, 0], [4, 2, 0], color=YELLOW, stroke_width=10, joint_type="bevel")
        tri3 = Polygon([6, 0, 0], [8, 0, 0], [7, 2, 0], color=BLUE, stroke_width=10, joint_type="round")
        labels = VGroup(
            Text("Miter", font_size=24).next_to(tri1, DOWN),
            Text("Bevel", font_size=24).next_to(tri2, DOWN),
            Text("Round", font_size=24).next_to(tri3, DOWN)
        )
        group = VGroup(tri1, tri2, tri3, labels).arrange(RIGHT, buff=1.5)
        self.play(Create(tri1), Create(tri2), Create(tri3), Write(labels))
        self.wait(2)
```