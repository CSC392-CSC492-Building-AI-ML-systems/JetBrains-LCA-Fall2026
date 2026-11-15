```python
from manim import *

class TitleAndEquationScene(Scene):
    def construct(self):
        title = Tex(r"Fourier Transform", font_size=60)
        equation = MathTex(
            r"\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} dx",
            font_size=48
        ).next_to(title, DOWN)
        self.play(Write(title), FadeIn(equation))
        new_title = Tex(r"Frequency Domain", font_size=60)
        self.play(Transform(title, new_title), FadeOut(equation))
        self.wait()
        grid = NumberPlane()
        grid_title = Tex("Original Grid", font_size=36).to_edge(UP)
        self.play(Create(grid), FadeIn(grid_title))
        self.wait()
        def nl_func(point):
            x, y, z = point
            return np.array([x, y + 0.3 * np.sin(x), z])
        self.play(grid.animate.apply_function(nl_func))
        new_grid_title = Tex("Non-linear Transformation", font_size=36).to_edge(UP)
        self.play(Transform(grid_title, new_grid_title))
        self.wait()

class SquareToCircleScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        circle = Circle()
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
        self.wait()

class SquarePointwiseFunctionScene(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        def pointwise_func(point):
            x, y, z = point
            return np.array([x, y + 0.2 * np.sin(3 * x), z])
        self.play(square.animate.apply_function(pointwise_func))
        self.wait()

class TextAndEquationScene(Scene):
    def construct(self):
        text = Text("Euler's Identity", font_size=48)
        equation = MathTex(r"e^{i\pi} + 1 = 0", font_size=60).next_to(text, DOWN)
        self.play(Write(text), Write(equation))
        self.wait()

class SquareAndDecimalUpdaterScene(Scene):
    def construct(self):
        square = Square().set_fill(BLUE, opacity=0.5).move_to(LEFT * 3)
        decimal = DecimalNumber(0, num_decimal_places=2, font_size=48)
        decimal.add_updater(lambda d: d.next_to(square, UP).set_value(square.get_center()[0]))
        self.add(square, decimal)
        self.play(square.animate.move_to(RIGHT * 3), run_time=4)
        self.wait()
        decimal.clear_updaters()

class SpiralShapesScene(Scene):
    def construct(self):
        shapes = VGroup(
            Square(),
            Circle(),
            Triangle(),
            RegularPolygon(5),
            Star(outer_radius=0.7, n=6)
        ).arrange(RIGHT, buff=0.7)
        pi = MathTex(r"\pi", font_size=144).move_to(UP * 2)
        all_mobs = VGroup(shapes, pi)
        for i, mob in enumerate(shapes):
            mob.move_to(ORIGIN)
            mob.set_opacity(0)
        pi.set_opacity(0)
        self.add(*shapes, pi)
        anims = []
        for i, mob in enumerate(shapes):
            angle = i * PI / 3
            target = RIGHT * (2 + i) + UP * np.sin(angle)
            anims.append(
                AnimationGroup(
                    mob.animate.move_to(target).set_opacity(1).rotate(angle),
                    lag_ratio=0.2
                )
            )
        anims.append(pi.animate.set_opacity(1))
        self.play(*anims, run_time=2)
        self.wait()
        self.play(FadeOut(all_mobs))
        self.wait()

class TrianglesWithLineJointsScene(Scene):
    def construct(self):
        tri1 = Polygon([-2, 1, 0], [-3, -1, 0], [-1, -1, 0], color=YELLOW, stroke_width=16, joint_type=LineJoint.ROUND)
        tri2 = Polygon([0, 1, 0], [-1, -1, 0], [1, -1, 0], color=GREEN, stroke_width=16, joint_type=LineJoint.BEVEL)
        tri3 = Polygon([2, 1, 0], [1, -1, 0], [3, -1, 0], color=RED, stroke_width=16, joint_type=LineJoint.MITER)
        labels = VGroup(
            Text("ROUND", font_size=24).next_to(tri1, DOWN),
            Text("BEVEL", font_size=24).next_to(tri2, DOWN),
            Text("MITER", font_size=24).next_to(tri3, DOWN)
        )
        self.play(Create(tri1), Create(tri2), Create(tri3))
        self.play(Write(labels))
        self.wait()
```