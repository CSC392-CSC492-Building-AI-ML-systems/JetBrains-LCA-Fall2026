```python
import pyvista as pv
from pyvista import examples

# Create a plotter
plotter = pv.Plotter(shape=(6, 5), window_size=(2000, 2400), border=False)

# 1. Supertoroid
supertoroid = pv.ParametricSuperToroid()
plotter.subplot(0, 0)
plotter.add_text("Supertoroid", font_size=10)
plotter.add_mesh(supertoroid, color="lightblue")

# 2. Ellipsoid
ellipsoid = pv.ParametricEllipsoid(1.0, 1.5, 2.0)
plotter.subplot(0, 1)
plotter.add_text("Ellipsoid", font_size=10)
plotter.add_mesh(ellipsoid, color="lightblue")

# 3. Partial Parametric Ellipsoid (plotting direction)
partial_ellipsoid = pv.ParametricEllipsoid(1.0, 1.5, 2.0)
partial_ellipsoid = partial_ellipsoid.clip(normal='x', origin=(0,0,0))
plotter.subplot(0, 2)
plotter.add_text("Partial Ellipsoid", font_size=10)
plotter.add_mesh(partial_ellipsoid, color="lightblue")

# 4. Pseudosphere
pseudosphere = pv.ParametricPseudosphere()
plotter.subplot(0, 3)
plotter.add_text("Pseudosphere", font_size=10)
plotter.add_mesh(pseudosphere, color="lightblue")

# 5. Bohemian Dome
bohemian = pv.ParametricBohemianDome()
plotter.subplot(0, 4)
plotter.add_text("Bohemian Dome", font_size=10)
plotter.add_mesh(bohemian, color="lightblue")

# 6. Bour
bour = pv.ParametricBour()
plotter.subplot(1, 0)
plotter.add_text("Bour", font_size=10)
plotter.add_mesh(bour, color="lightblue")

# 7. Boy's Surface
boys = pv.ParametricBoy()
plotter.subplot(1, 1)
plotter.add_text("Boy's Surface", font_size=10)
plotter.add_mesh(boys, color="lightblue")

# 8. Catalan Minimal
catalan = pv.ParametricCatalanMinimal()
plotter.subplot(1, 2)
plotter.add_text("Catalan Minimal", font_size=10)
plotter.add_mesh(catalan, color="lightblue")

# 9. Conic Spiral
conic_spiral = pv.ParametricConicSpiral()
plotter.subplot(1, 3)
plotter.add_text("Conic Spiral", font_size=10)
plotter.add_mesh(conic_spiral, color="lightblue")

# 10. Cross Cap
cross_cap = pv.ParametricCrossCap()
plotter.subplot(1, 4)
plotter.add_text("Cross Cap", font_size=10)
plotter.add_mesh(cross_cap, color="lightblue")

# 11. Dini
dini = pv.ParametricDini()
plotter.subplot(2, 0)
plotter.add_text("Dini", font_size=10)
plotter.add_mesh(dini, color="lightblue")

# 12. Enneper (plotting position "yz")
enneper = pv.ParametricEnneper()
plotter.subplot(2, 1)
plotter.add_text("Enneper (yz)", font_size=10)
plotter.add_mesh(enneper, color="lightblue")
plotter.view_yz()

# 13. Figure-8 Klein
figure8_klein = pv.ParametricFigure8Klein()
plotter.subplot(2, 2)
plotter.add_text("Figure-8 Klein", font_size=10)
plotter.add_mesh(figure8_klein, color="lightblue")

# 14. Henneberg
henneberg = pv.ParametricHenneberg()
plotter.subplot(2, 3)
plotter.add_text("Henneberg", font_size=10)
plotter.add_mesh(henneberg, color="lightblue")

# 15. Klein
klein = pv.ParametricKlein()
plotter.subplot(2, 4)
plotter.add_text("Klein", font_size=10)
plotter.add_mesh(klein, color="lightblue")

# 16. Kuen
kuen = pv.ParametricKuen()
plotter.subplot(3, 0)
plotter.add_text("Kuen", font_size=10)
plotter.add_mesh(kuen, color="lightblue")

# 17. Mobius
mobius = pv.ParametricMobius()
plotter.subplot(3, 1)
plotter.add_text("Mobius", font_size=10)
plotter.add_mesh(mobius, color="lightblue")

# 18. Plucker Conoid
plucker = pv.ParametricPluckerConoid()
plotter.subplot(3, 2)
plotter.add_text("Plucker Conoid", font_size=10)
plotter.add_mesh(plucker, color="lightblue")

# 19. Random Hills
random_hills = pv.ParametricRandomHills()
plotter.subplot(3, 3)
plotter.add_text("Random Hills", font_size=10)
plotter.add_mesh(random_hills, color="lightblue")

# 20. Roman
roman = pv.ParametricRoman()
plotter.subplot(3, 4)
plotter.add_text("Roman", font_size=10)
plotter.add_mesh(roman, color="lightblue")

# 21. Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid()
plotter.subplot(4, 0)
plotter.add_text("Super Ellipsoid", font_size=10)
plotter.add_mesh(super_ellipsoid, color="lightblue")

# 22. Torus
torus = pv.ParametricTorus()
plotter.subplot(4, 1)
plotter.add_text("Torus", font_size=10)
plotter.add_mesh(torus, color="lightblue")

# 23. Circular Arc (define points and center)
pointa = [1, 0, 0]
pointb = [0, 1, 0]
center = [0, 0, 0]
arc = pv.plot_over_circular_arc(pointa, pointb, center=center, resolution=100, show=False)
plotter.subplot(4, 2)
plotter.add_text("Circular Arc", font_size=10)
plotter.add_mesh(arc, color="lightblue", line_width=4)

# 24. Extruded Half Arc (extrude in z, show edges)
half_arc = pv.CircularArc(pointa, pointb, center=center, resolution=100)
extruded = half_arc.extrude([0, 0, 1])
plotter.subplot(4, 3)
plotter.add_text("Extruded Half Arc", font_size=10)
plotter.add_mesh(extruded, color="lightblue", show_edges=True)

# Hide the last subplot (empty)
plotter.subplot(4, 4)
plotter.hide_axes()

# Show the plotter
plotter.show()
```