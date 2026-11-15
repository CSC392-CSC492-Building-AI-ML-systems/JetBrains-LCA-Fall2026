```python
import os
import sys
from DDG4 import DDG4Kernel, Geant4, ParticleGun, setupPhysics, setupDetector

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    kernel = DDG4Kernel()
    
    # Load geometry from XML file
    install_dir = os.environ.get('DD4hepExamplesINSTALL')
    if not install_dir:
        raise EnvironmentError("DD4hepExamplesINSTALL environment variable not set.")
    geometry_path = os.path.join(install_dir, 'OpticalSurfaces', 'compact', 'OpticalSurfaces.xml')
    kernel.loadGeometry(geometry_path)
    
    # Import constants from detector description
    kernel.detectorDescription().importConstants()
    
    # Configure Geant4 instance
    g4 = Geant4(kernel)
    g4.setupTrackingField()
    g4.setupEventActions()
    g4.setupDetectorConstruction()
    
    # Setup particle gun
    gun = ParticleGun()
    gun.particle = "gamma"
    gun.energy = "5 keV"
    gun.multiplicity = 1
    g4.addPrimaryGenerator(gun)
    
    # Setup tracker
    g4.addTracker("MaterialTester")
    
    # Setup physics list
    setupPhysics(g4, "QGSP_BERT")
    
    # Setup UI with macro if provided
    macro = sys.argv[1] if len(sys.argv) > 1 else None
    if macro:
        g4.setupUI(macro)
    else:
        g4.setupUI()
    
    # Execute Geant4 instance
    g4.execute()

if __name__ == "__main__":
    setup_dd4hep_simulation()
```