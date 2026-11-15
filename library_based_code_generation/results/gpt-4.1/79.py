```python
import os
import sys
from DDG4 import Kernel, Geant4, ParticleGun, Tracker, PhysicsList

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    kernel = Kernel()
    
    # Load geometry from XML file
    dd4hep_install = os.environ.get('DD4hepExamplesINSTALL')
    if not dd4hep_install:
        raise EnvironmentError("DD4hepExamplesINSTALL environment variable not set.")
    geometry_path = os.path.join(dd4hep_install, 'OpticalSurfaces', 'compact', 'OpticalSurfaces.xml')
    kernel.loadGeometry(geometry_path)
    
    # Import constants from detector description
    kernel.detectorDescription().importConstants()
    
    # Configure Geant4 instance
    g4 = Geant4(kernel)
    g4.trackingField = True
    g4.eventActions = True
    g4.detectorConstruction = True
    
    # Set up particle gun
    gun = ParticleGun()
    gun.particle = "gamma"
    gun.energy = "5 keV"
    gun.multiplicity = 1
    g4.addAction(gun)
    
    # Set up tracker
    tracker = Tracker("MaterialTester")
    g4.addAction(tracker)
    
    # Set up physics list
    physics = PhysicsList("QGSP_BERT")
    g4.addAction(physics)
    
    # Setup UI and macro
    macro = sys.argv[1] if len(sys.argv) > 1 else None
    if macro:
        g4.ui = True
        g4.macro = macro
    else:
        g4.ui = True
    
    # Execute Geant4 instance
    g4.run()

if __name__ == "__main__":
    setup_dd4hep_simulation()
```