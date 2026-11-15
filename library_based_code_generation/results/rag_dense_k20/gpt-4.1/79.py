```python
import os
import sys
from DDG4 import DDG4Kernel, Geant4

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
    gun = g4.setupGun()
    gun.particle = "gamma"
    gun.energy = "5 keV"
    gun.multiplicity = 1
    
    # Setup tracker
    g4.setupDetectors(['MaterialTester'])
    
    # Setup physics list
    g4.setupPhysics('QGSP_BERT')
    
    # Setup UI with macro if provided
    macro = sys.argv[1] if len(sys.argv) > 1 else None
    g4.setupUI(macro)
    
    # Execute Geant4 instance
    g4.execute()

if __name__ == "__main__":
    setup_dd4hep_simulation()
```