```python
import os
import dd4hep
from DDG4 import DD4hepSimulation, Geant4, enableDetailedHitsAndParticleInfo

# Initialize DD4hep kernel
kernel = dd4hep.DDKernel.Instance()

# Load geometry from XML file in DD4hepExamplesINSTALL
geometry_path = os.path.join(os.environ['DD4hepExamplesINSTALL'], 'DDDetectors/compact/Tracker.xml')
kernel.loadGeometry(geometry_path)

# Import constants from detector description
kernel.detector().importConstants(kernel.detectorDescription())

# Enable detailed hits and particle info
enableDetailedHitsAndParticleInfo(kernel)

# Set up Geant4 simulation instance
sim = DD4hepSimulation()
g4 = Geant4(kernel)
sim.setGeant4(g4)

# Set up user interface
g4.setupUserInterface()

# Configure tracking field
g4.trackingField = True

# Configure event actions
g4.eventActions = True

# Set up particle gun: gamma, 5 keV, multiplicity 1
g4.setupGun(particle='gamma', energy=5e-3, multiplicity=1)

# Set up tracker
g4.setupTracker()

# Set up physics list and add particle groups and processes
physics = g4.setupPhysics('FTFP_BERT')
physics.addParticleGroup('gamma')
physics.addParticleGroup('e-')
physics.addParticleGroup('e+')
physics.addProcess('phot')
physics.addProcess('compt')
physics.addProcess('conv')

# Execute Geant4 instance
g4.execute()
```