```python
from manim import *

class Scene1(Scene):
    def construct(self):
        # Display LaTeX title and equation
        title = Tex(r"Manim Scene Example").to_edge(UP)
        equation = MathTex(r"E=mc^2").next_to(title, DOWN)
        self.play(Write(title), Write(equation))
        self.wait(1)
        # Transform the title and fade out the equation
        new_title = Tex(r"Transformed Title").to_edge(UP)
        self.play(Transform(title, new_title), FadeOut(equation))
        self.wait(1)
        # Create a grid and display a title
        grid = NumberPlane()
        grid_title = Tex("Grid Example").to_edge(DOWN)
        self.play(Create(grid), Write(grid_title))
        self.wait(1)
        # Apply a non-linear function to the grid
        self.play(grid.animate.apply_function(
            lambda p: [p[0], p[1] + 0.2 * np.sin(p[0]), p[2]]
        ))
        self.wait(1)
        # Transform the grid title
        new_grid_title = Tex("Non-linear Grid").to_edge(DOWN)
        self.play(Transform(grid_title, new_grid_title))
        self.wait(1)
        self.play(FadeOut(grid), FadeOut(grid_title), FadeOut(title))

class Scene2(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(0.5)
        circle = Circle()
        self.play(Transform(square, circle))
        self.wait(0.5)
        self.play(FadeOut(square))

class Scene3(Scene):
    def construct(self):
        square = Square()
        self.play(Create(square))
        self.wait(0.5)
        # Apply a pointwise function to the square
        self.play(square.animate.apply_function(
            lambda p: [p[0] + 0.3 * np.sin(p[1]), p[1], p[2]]
        ))
        self.wait(1)
        self.play(FadeOut(square))

class Scene4(Scene):
    def construct(self):
        text = Text("Displaying Math:")
        equation = MathTex(r"\int_a^b f(x)\,dx")
        group = VGroup(text, equation).arrange(DOWN)
        self.play(Write(group))
        self.wait(2)
        self.play(FadeOut(group))

class Scene5(Scene):
    def construct(self):
        square = Square().set_color(BLUE).move_to(LEFT * 3)
        decimal = DecimalNumber(0, num_decimal_places=2)
        decimal.add_updater(lambda d: d.next_to(square, UP).set_value(square.get_center()[0]))
        self.add(square, decimal)
        self.play(square.animate.move_to(RIGHT * 3), run_time=3)
        self.wait(1)
        self.play(FadeOut(square), FadeOut(decimal))

class Scene6(Scene):
    def construct(self):
        shapes = VGroup(
            Square().shift(LEFT + UP),
            Circle().shift(RIGHT + UP),
            Triangle().shift(LEFT + DOWN),
            RegularPolygon(5).shift(RIGHT + DOWN)
        )
        pi = MathTex(r"\pi").scale(3)
        self.play(*[FadeIn(s, scale=0.5) for s in shapes], FadeIn(pi))
        self.wait(1)
        # Spiral in the shapes and pi
        def spiral(mob, i):
            return mob.animate.move_to(ORIGIN).rotate(2 * PI * (i+1) / len(shapes))
        self.play(
            *[spiral(s, i) for i, s in enumerate(shapes)],
            pi.animate.move_to(ORIGIN).scale(0.5),
            run_time=2
        )
        self.wait(1)
        self.play(FadeOut(shapes), FadeOut(pi))

class Scene7(Scene):
    def construct(self):
        # Miter joint
        tri1 = Polygon([0,0,0], [2,0,0], [1,2,0], color=RED, stroke_width=20, joint_type="miter").shift(LEFT*3)
        # Bevel joint
        tri2 = Polygon([0,0,0], [2,0,0], [1,2,0], color=GREEN, stroke_width=20, joint_type="bevel")
        # Round joint
        tri3 = Polygon([0,0,0], [2,0,0], [1,2,0], color=BLUE, stroke_width=20, joint_type="round").shift(RIGHT*3)
        self.play(Create(tri1), Create(tri2), Create(tri3))
        self.wait(2)
        self.play(FadeOut(tri1), FadeOut(tri2), FadeOut(tri3))
```