```python
import os
import dd4hep
from dd4hep import Detector
from dd4hep.simulation import Geant4, setupTracker, setupUserParticleHandler
from dd4hep.simulation import ParticleGun, PhysicsList, EventAction, TrackingField, UserInterface

# Initialize kernel and load geometry
install_dir = os.environ.get('DD4hepExamplesINSTALL')
geometry_xml = os.path.join(install_dir, 'DDDetectors/compact/Tracker.xml')
kernel = dd4hep.DDKernel()
detector = Detector.getInstance()
detector.fromXML(geometry_xml)

# Import constants from detector description
detector.importConstants()

# Set up Geant4 instance with tracker
g4 = Geant4(kernel)
setupTracker(g4, detector)

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
particle_gun.particle = "gamma"
particle_gun.energy = 5 * dd4hep.keV
particle_gun.multiplicity = 1
g4.setPrimaryGenerator(particle_gun)

# Set up tracker (already done above, but ensure configuration)
setupTracker(g4, detector)

# Set up physics list
physics = PhysicsList("FTFP_BERT")
physics.addParticleGroup("gamma")
physics.addParticleGroup("e-")
physics.addParticleGroup("e+")
physics.addParticleGroup("proton")
physics.addProcess("photoelectric")
physics.addProcess("compton")
physics.addProcess("gammaConversion")
physics.addProcess("ionisation")
physics.addProcess("bremsstrahlung")
g4.setPhysicsList(physics)

# Execute Geant4 instance
g4.run()
```