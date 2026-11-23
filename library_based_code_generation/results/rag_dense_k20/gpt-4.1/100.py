```python
import os
import dd4hep
from dd4hep import Detector
from DDG4 import Geant4, setupGun, setupPhysics, setupUserParticleHandler, setupUserPhysics

# Initialize DD4hep kernel
kernel = dd4hep.DDKernel.instance()

# Load geometry from XML file in DD4hepExamplesINSTALL
geometry_path = os.path.join(os.environ['DD4hepExamplesINSTALL'], 'DDDetectors/compact/Tracker.xml')
kernel.loadGeometry(geometry_path)

# Import constants from detector description
detector = kernel.detectorDescription()
detector.importConstants(kernel)

# Set up Geant4 simulation instance with tracker
g4 = Geant4(kernel)
g4.setupTrackingField()
g4.setupUserInterface()
g4.setupEventActions()

# Set up particle gun: gamma, 5 keV, multiplicity 1
setupGun(g4, particle='gamma', energy=5e-3, multiplicity=1)

# Set up tracker
g4.setupTracker()

# Set up physics list, add particle groups and processes
setupPhysics(g4, physicsList='FTFP_BERT')
setupUserParticleHandler(g4)
setupUserPhysics(g4)

# Execute Geant4 simulation
g4.run()
```