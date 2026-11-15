```python
import pyvista as pv
import numpy as np

# Create a plotter
plotter = pv.Plotter(window_size=[1800, 1200])

# 1. Supertoroid
supertoroid = pv.ParametricSuperToroid()
plotter.add_mesh(supertoroid, color='lightblue', show_edges=False, label='Supertoroid')

# 2. Ellipsoid
ellipsoid = pv.ParametricEllipsoid()
plotter.add_mesh(ellipsoid, color='lightblue', show_edges=False, label='Ellipsoid')

# 3. Partial Parametric Ellipsoid (use direction)
partial_ellipsoid = pv.ParametricEllipsoid()
partial_ellipsoid = partial_ellipsoid.clip(normal='z', origin=(0,0,0))
plotter.add_mesh(partial_ellipsoid, color='lightblue', show_edges=False, label='Partial Ellipsoid')

# 4. Pseudosphere
pseudosphere = pv.ParametricPseudosphere()
plotter.add_mesh(pseudosphere, color='lightblue', show_edges=False, label='Pseudosphere')

# 5. Bohemian Dome
bohemian = pv.ParametricBohemianDome()
plotter.add_mesh(bohemian, color='lightblue', show_edges=False, label='Bohemian Dome')

# 6. Bour
bour = pv.ParametricBour()
plotter.add_mesh(bour, color='lightblue', show_edges=False, label='Bour')

# 7. Boy's Surface
boys = pv.ParametricBoy()
plotter.add_mesh(boys, color='lightblue', show_edges=False, label="Boy's Surface")

# 8. Catalan Minimal
catalan = pv.ParametricCatalanMinimal()
plotter.add_mesh(catalan, color='lightblue', show_edges=False, label='Catalan Minimal')

# 9. Conic Spiral
conic_spiral = pv.ParametricConicSpiral()
plotter.add_mesh(conic_spiral, color='lightblue', show_edges=False, label='Conic Spiral')

# 10. Cross Cap
cross_cap = pv.ParametricCrossCap()
plotter.add_mesh(cross_cap, color='lightblue', show_edges=False, label='Cross Cap')

# 11. Dini
dini = pv.ParametricDini()
plotter.add_mesh(dini, color='lightblue', show_edges=False, label='Dini')

# 12. Enneper (plotting position "yz")
enneper = pv.ParametricEnneper()
plotter.add_mesh(enneper, color='lightblue', show_edges=False, label='Enneper')
plotter.camera_position = 'yz'

# 13. Figure-8 Klein
figure8_klein = pv.ParametricFigure8Klein()
plotter.add_mesh(figure8_klein, color='lightblue', show_edges=False, label='Figure-8 Klein')

# 14. Henneberg
henneberg = pv.ParametricHenneberg()
plotter.add_mesh(henneberg, color='lightblue', show_edges=False, label='Henneberg')

# 15. Klein
klein = pv.ParametricKlein()
plotter.add_mesh(klein, color='lightblue', show_edges=False, label='Klein')

# 16. Kuen
kuen = pv.ParametricKuen()
plotter.add_mesh(kuen, color='lightblue', show_edges=False, label='Kuen')

# 17. Mobius
mobius = pv.ParametricMobius()
plotter.add_mesh(mobius, color='lightblue', show_edges=False, label='Mobius')

# 18. Plucker Conoid
plucker = pv.ParametricPluckerConoid()
plotter.add_mesh(plucker, color='lightblue', show_edges=False, label='Plucker Conoid')

# 19. Random Hills
random_hills = pv.ParametricRandomHills()
plotter.add_mesh(random_hills, color='lightblue', show_edges=False, label='Random Hills')

# 20. Roman
roman = pv.ParametricRoman()
plotter.add_mesh(roman, color='lightblue', show_edges=False, label='Roman')

# 21. Super Ellipsoid
super_ellipsoid = pv.ParametricSuperEllipsoid()
plotter.add_mesh(super_ellipsoid, color='lightblue', show_edges=False, label='Super Ellipsoid')

# 22. Torus
torus = pv.ParametricTorus()
plotter.add_mesh(torus, color='lightblue', show_edges=False, label='Torus')

# 23. Circular Arc (define points and center)
pointa = [1, 0, 0]
pointb = [0, 1, 0]
center = [0, 0, 0]
arc = pv.CircularArc(pointa, pointb, center)
plotter.add_mesh(arc, color='lightblue', line_width=5, label='Circular Arc')

# 24. Extruded Half Arc (extrude in z, show edges)
half_arc = pv.CircularArc([1, 0, 0], [0, 1, 0], [0, 0, 0], angle=90)
extruded = half_arc.extrude([0, 0, 1])
plotter.add_mesh(extruded, color='lightblue', show_edges=True, label='Extruded Half Arc')

plotter.add_legend()
plotter.show()
```