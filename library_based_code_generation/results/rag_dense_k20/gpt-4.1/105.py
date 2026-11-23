```python
import pyvista as pv
from pyvista import examples
import numpy as np

# Create a plotter
plotter = pv.Plotter(shape=(5, 5), window_size=(2000, 2000))

# 1. Supertoroid
supertoroid = pv.ParametricSuperToroid()
plotter.subplot(0, 0)
plotter.add_mesh(supertoroid, color='lightblue')
plotter.add_text("Supertoroid")

# 2. Ellipsoid
ellipsoid = pv.ParametricEllipsoid(1.0, 1.5, 2.0)
plotter.subplot(0, 1)
plotter.add_mesh(ellipsoid, color='lightblue')
plotter.add_text("Ellipsoid")

# 3. Partial Parametric Ellipsoid (plotting direction)
partial_ellipsoid = pv.ParametricEllipsoid(1.0, 1.5, 2.0)
partial_ellipsoid = partial_ellipsoid.clip(normal='x', origin=(0,0,0))
plotter.subplot(0, 2)
plotter.add_mesh(partial_ellipsoid, color='lightblue')
plotter.add_text("Partial Ellipsoid")

# 4. Pseudosphere
pseudosphere = pv.ParametricPseudosphere()
plotter.subplot(0, 3)
plotter.add_mesh(pseudosphere, color='lightblue')
plotter.add_text("Pseudosphere")

# 5. Bohemian Dome
bohemian = pv.ParametricBohemianDome()
plotter.subplot(0, 4)
plotter.add_mesh(bohemian, color='lightblue')
plotter.add_text("Bohemian Dome")

# 6. Bour
bour = pv.ParametricBour()
plotter.subplot(1, 0)
plotter.add_mesh(bour, color='lightblue')
plotter.add_text("Bour")

# 7. Boy's Surface
boys = pv.ParametricBoy()
plotter.subplot(1, 1)
plotter.add_mesh(boys, color='lightblue')
plotter.add_text("Boy's Surface")

# 8. Catalan Minimal
catalan = pv.ParametricCatalanMinimal()
plotter.subplot(1, 2)
plotter.add_mesh(catalan, color='lightblue')
plotter.add_text("Catalan Minimal")

# 9. Conic Spiral
conic_spiral = pv.ParametricConicSpiral()
plotter.subplot(1, 3)
plotter.add_mesh(conic_spiral, color='lightblue')
plotter.add_text("Conic Spiral")

# 10. Cross Cap
cross_cap = pv.ParametricCrossCap()
plotter.subplot(1, 4)
plotter.add_mesh(cross_cap, color='lightblue')
plotter.add_text("Cross Cap")

# 11. Dini
dini = pv.ParametricDini()
plotter.subplot(2, 0)
plotter.add_mesh(dini, color='lightblue')
plotter.add_text("Dini")

# 12. Enneper (plotting position "yz")
enneper = pv.ParametricEnneper()
enneper.rotate_x(90)
plotter.subplot(2, 1)
plotter.add_mesh(enneper, color='lightblue')
plotter.add_text("Enneper (yz)")

# 13. Figure-8 Klein
figure8_klein = pv.ParametricFigure8Klein()
plotter.subplot(2, 2)
plotter.add_mesh(figure8_klein, color='lightblue')
plotter.add_text("Figure-8 Klein")

# 14. Henneberg
henneberg = pv.ParametricHenneberg()
plotter.subplot(2, 3)
plotter.add_mesh(henneberg, color='lightblue')
plotter.add_text("Henneberg")

# 15. Klein
klein = pv.ParametricKlein()
plotter.subplot(2, 4)
plotter.add_mesh(klein, color='lightblue')
plotter.add_text("Klein")

# 16. Kuen
kuen = pv.ParametricKuen()
plotter.subplot(3, 0)
plotter.add_mesh(kuen, color='lightblue')
plotter.add_text("Kuen")

# 17. Mobius
mobius = pv.ParametricMobius()
plotter.subplot(3, 1)
plotter.add_mesh(mobius, color='lightblue')
plotter.add_text("Mobius")

# 18. Plucker Conoid
plucker = pv.ParametricPluckerConoid()
plotter.subplot(3, 2)
plotter.add_mesh(plucker, color='lightblue')
plotter.add_text("Plucker Conoid")

# 19. Random Hills
random_hills = pv.ParametricRandomHills()
plotter.subplot(3, 3)
plotter.add_mesh(random_hills, color='lightblue')
plotter.add_text("Random Hills")

# 20. Roman
roman = pv.ParametricRoman()
plotter.subplot(3, 4)
plotter.add_mesh(roman, color='lightblue')
plotter.add_text("Roman")

# 21. Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid()
plotter.subplot(4, 0)
plotter.add_mesh(super_ellipsoid, color='lightblue')
plotter.add_text("Super Ellipsoid")

# 22. Torus
torus = pv.ParametricTorus()
plotter.subplot(4, 1)
plotter.add_mesh(torus, color='lightblue')
plotter.add_text("Torus")

# 23. Circular Arc (define points and center)
center = np.array([0, 0, 0])
pointa = np.array([1, 0, 0])
pointb = np.array([0, 1, 0])
arc = pv.CircularArc(pointa, pointb, center)
plotter.subplot(4, 2)
plotter.add_mesh(arc, color='lightblue', line_width=5)
plotter.add_text("Circular Arc")

# 24. Extruded Half Arc (extrude in z, show edges)
half_arc = pv.CircularArc([1, 0, 0], [0, 1, 0], [0, 0, 0], angle=90)
extruded = half_arc.extrude([0, 0, 1])
plotter.subplot(4, 3)
plotter.add_mesh(extruded, color='lightblue', show_edges=True)
plotter.add_text("Extruded Half Arc")

# Hide axes for clarity
for i in range(5):
    for j in range(5):
        plotter.subplot(i, j)
        plotter.hide_axes()

plotter.show()
```