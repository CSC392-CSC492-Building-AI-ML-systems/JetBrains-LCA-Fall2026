```python
import os
import dd4hep
from dd4hep import Detector
from dd4hep.simulation import Kernel
from dd4hep.simulation import Geant4
from dd4hep.simulation import geant4
from dd4hep.simulation import ParticleGun
from dd4hep.simulation import PhysicsList
from dd4hep.simulation import EventAction
from dd4hep.simulation import TrackingField
from dd4hep.simulation import Tracker
from dd4hep.simulation import UserInterface

# Initialize kernel
kernel = Kernel()

# Load geometry from XML file in DD4hepExamplesINSTALL
install_dir = os.environ.get('DD4hepExamplesINSTALL')
geometry_file = os.path.join(install_dir, 'DDDetectors/compact/Tracker.xml')
detector = Detector.getInstance()
detector.fromXML(geometry_file)

# Import constants from detector description
detector.importConstants(detector.description())

# Set up Geant4 instance with tracker
g4 = Geant4(kernel)
tracker = Tracker()
g4.addDetector(tracker)

# Configure user interface
ui = UserInterface()
g4.setUserInterface(ui)

# Configure tracking field
field = TrackingField()
g4.setTrackingField(field)

# Configure event actions
event_action = EventAction()
g4.setEventAction(event_action)

# Set up particle gun
particle_gun = ParticleGun()
particle_gun.particleName = "gamma"
particle_gun.energy = 5 * dd4hep.keV
particle_gun.multiplicity = 1
g4.setPrimaryGenerator(particle_gun)

# Set up tracker (already added above, but can configure further if needed)
# tracker.configure(...)

# Set up physics list
physics_list = PhysicsList("FTFP_BERT")
physics_list.addParticleGroup("gamma")
physics_list.addParticleGroup("e-")
physics_list.addParticleGroup("e+")
physics_list.addParticleGroup("proton")
physics_list.addProcess("photoelectric")
physics_list.addProcess("compton")
physics_list.addProcess("gammaConversion")
physics_list.addProcess("ionisation")
physics_list.addProcess("bremsstrahlung")
g4.setPhysicsList(physics_list)

# Execute Geant4 instance
g4.run()
```