```python
import dream
import dream.settings
import dream.enums as enums

ds = dream.settings.Scenario()

# Radial grid
ds.radialgrid.setB0(5)
ds.radialgrid.setMinorRadius(0.22)
ds.radialgrid.setWallRadius(0.22)
ds.radialgrid.setNr(1)

# Ion species: fully ionized D
ds.eqsys.ni.addIon(name='D', Z=1, n=5e19, iontype=dream.settings.ions.IONS_DYNAMIC_FULLY_IONIZED)

# Electron temperature and density
ds.eqsys.T_cold.setPrescribedData(1e3)
ds.eqsys.n_cold.setPrescribedData(5e19)

# Electric field
ds.eqsys.E_field.setPrescribedData(0.6)

# Initial runaway electron density profile
ds.eqsys.n_re.setInitialProfile(1e15)

# Collision frequency mode: ultra-relativistic
ds.collisions.collfreq_mode = enums.Collisions.COLLFREQ_MODE_ULTRAREL

# Hot-tail grid disabled
ds.hottailgrid.setEnabled(False)

# Runaway electron grid (if enabled)
if ds.runawaygrid.enabled:
    ds.runawaygrid.setNr(50)
    ds.runawaygrid.setNp(100)
    ds.runawaygrid.setPmax(0.5)
    ds.runawaygrid.setAdvectionInterpolationMethod(enums.AdvectionInterpolation.FLUXLIMITER)
    ds.runawaygrid.setInitializationType(enums.RunawayElectronDistribution.INITIALIZATION_ISOTROPIC)

# Dreicer and avalanche settings
ds.eqsys.n_re.setDreicer(enums.DreicerRate.NN)
ds.eqsys.n_re.setAvalanche(enums.AvalancheMode.FLUID)

# Advection interpolation method: flux limiter
ds.eqsys.n_re.setAdvectionInterpolationMethod(enums.AdvectionInterpolation.FLUXLIMITER)

# Initialization method: isotropic
ds.eqsys.n_re.setInitializationType(enums.RunawayElectronDistribution.INITIALIZATION_ISOTROPIC)

# Solver: nonlinear, verbose, relative tolerance for j_re
ds.solver.setType(enums.Solver.NONLINEAR)
ds.solver.setVerbose(True)
ds.solver.setTolerance(1e-4, j_re=True)

# Fluid effects
ds.eqsys.n_re.setFluid(True)

# Time stepper
ds.timestep.setTmax(1e-1)
ds.timestep.setNt(20)

# Save settings to HDF5
ds.save('dream_settings.h5')
```