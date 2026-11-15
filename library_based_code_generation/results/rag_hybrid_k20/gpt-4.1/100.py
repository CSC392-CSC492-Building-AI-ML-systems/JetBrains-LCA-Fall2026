```python
import os
import dd4hep
from dd4hep.simulation import Geant4, DD4hepSimulation, setupGun, setupTracker, setupPhysics
from dd4hep.simulation import enableDetailedHitsAndParticleInfo, setDumpDetailedParticleInfo
from dd4hep.simulation import setupUserParticleHandler, setupUserPhysics
from dd4hep.simulation import forceDD4HEP, loadDDG4, import_geometry
from dd4hep.simulation import commandsInitialize

# Initialize DD4hep kernel
kernel = dd4hep.DDKernel.instance()

# Load geometry from XML file in DD4hepExamplesINSTALL
geometry_path = os.path.join(os.environ['DD4hepExamplesINSTALL'], 'DDDetectors/compact/Tracker.xml')
import_geometry(kernel, geometry_path)

# Import constants from detector description
detector = kernel.detectorDescription()
detector.importConstants()

# Set up Geant4 simulation instance
sim = Geant4(kernel)

# Enable detailed hits and particle info
enableDetailedHitsAndParticleInfo(sim)
setDumpDetailedParticleInfo(sim, True)

# Configure user interface, tracking field, and event actions
sim.setupUserInterface()
sim.setupTrackingField()
sim.setupEventActions()

# Set up particle gun: gamma, 5 keV, multiplicity 1
setupGun(sim, particle='gamma', energy=5e-3, multiplicity=1)

# Set up tracker
setupTracker(sim)

# Set up physics list and add particle groups/processes
physics = setupPhysics(sim, physics_list='FTFP_BERT')
physics.addParticleGroup('gamma')
physics.addParticleGroup('e-')
physics.addParticleGroup('e+')
physics.addProcess('phot')
physics.addProcess('compt')
physics.addProcess('conv')

# Execute Geant4 simulation
sim.execute()
```