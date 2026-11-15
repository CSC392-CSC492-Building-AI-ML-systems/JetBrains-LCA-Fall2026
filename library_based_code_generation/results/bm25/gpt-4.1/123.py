```python
from manim import *

class Scene1(ThreeDScene):
    def construct(self):
        # Display LaTeX title and equation
        title = Tex(r"Manim Scenes", font_size=60)
        equation = MathTex(r"E=mc^2", font_size=48).next_to(title, DOWN)
        self.play(Write(title), FadeIn(equation))
        self.wait(1)
        # Transform title and fade out equation
        new_title = Tex(r"Transformed Title", font_size=60)
        self.play(Transform(title, new_title), FadeOut(equation))
        self.wait(1)
        # Create grid and title
        grid = NumberPlane(x_range=[-4, 4, 1], y_range=[-3, 3, 1], background_line_style={"stroke_opacity": 0.5})
        grid_title = Tex("Grid", font_size=48).to_edge(UP)
        self.play(FadeIn(grid), Write(grid_title))
        self.wait(1)
        # Apply non-linear function to grid
        self.play(grid.animate.apply_function(lambda p: [p[0], p[1] + 0.3 * np.sin(p[0]), p[2]]))
        self.wait(1)
        # Transform grid title
        new_grid_title = Tex("Non-linear Grid", font_size=48).to_edge(UP)
        self.play(Transform(grid_title, new_grid_title))
        self.wait(1)
        self.play(FadeOut(grid), FadeOut(grid_title), FadeOut(title))

class Scene2(Scene):
    def construct(self):
        square = Square()
        circle = Circle()
        self.play(Create(square))
        self.wait(0.5)
        self.play(Transform(square, circle))
        self.wait(0.5)
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(0.5)
        # Apply pointwise function (e.g., wave)
        self.play(square.animate.apply_function(lambda p: [p[0], p[1] + 0.2 * np.sin(2 * p[0]), p[2]]))
        self.wait(1)
        self.play(FadeOut(square))

class Scene4(Scene):
    def construct(self):
        text = Text("Here is an equation:", font_size=36)
        eq = MathTex(r"\int_a^b f(x)\,dx = F(b) - F(a)", font_size=48).next_to(text, DOWN)
        self.play(Write(text), Write(eq))
        self.wait(2)
        self.play(FadeOut(text), FadeOut(eq))

class Scene5(Scene):
    def construct(self):
        square = Square().set_color(BLUE).move_to(LEFT * 3)
        decimal = DecimalNumber(0, num_decimal_places=2, font_size=36)
        decimal.add_updater(lambda d: d.next_to(square, UP).set_value(square.get_center()[0]))
        self.add(square, decimal)
        self.play(square.animate.move_to(RIGHT * 3), run_time=3)
        self.wait(0.5)
        self.play(FadeOut(square), FadeOut(decimal))

class Scene6(Scene):
    def construct(self):
        shapes = VGroup(
            Square().shift(LEFT + UP),
            Circle().shift(RIGHT + UP),
            Triangle().shift(LEFT + DOWN),
            RegularPolygon(5).shift(RIGHT + DOWN)
        )
        pi = MathTex(r"\pi", font_size=144)
        self.play(*[FadeIn(s, scale=0.5) for s in shapes], FadeIn(pi))
        self.wait(0.5)
        # Spiral in
        for i, s in enumerate(shapes):
            self.play(s.animate.shift(ORIGIN - s.get_center()).rotate(PI * (i+1)), run_time=0.5)
        self.play(pi.animate.scale(1.2).set_color(YELLOW), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(shapes), FadeOut(pi))

class Scene7(Scene):
    def construct(self):
        # Triangle with miter joint
        tri1 = Polygon([0,0,0], [2,0,0], [1,2,0], color=RED, stroke_width=12, joint_type="miter").shift(LEFT*3)
        # Triangle with round joint
        tri2 = Polygon([0,0,0], [2,0,0], [1,2,0], color=GREEN, stroke_width=12, joint_type="round")
        # Triangle with bevel joint
        tri3 = Polygon([0,0,0], [2,0,0], [1,2,0], color=BLUE, stroke_width=12, joint_type="bevel").shift(RIGHT*3)
        self.play(Create(tri1), Create(tri2), Create(tri3))
        self.wait(2)
```