```python
import os
import sys
from DDG4 import Kernel, Geant4, Gun, CommandLine

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    kernel = Kernel()
    
    # Load geometry from XML file
    install_dir = os.environ.get('DD4hepExamplesINSTALL')
    if not install_dir:
        raise RuntimeError("Environment variable 'DD4hepExamplesINSTALL' is not set.")
    geometry_path = os.path.join(install_dir, 'OpticalSurfaces', 'compact', 'OpticalSurfaces.xml')
    kernel.loadGeometry(geometry_path)
    
    # Import constants from detector description
    kernel.detectorDescription().importConstants()
    
    # Configure Geant4 instance
    g4 = Geant4(kernel)
    g4.setOption('UI', True)
    g4.setOption('trackingField', True)
    g4.setOption('eventActions', True)
    
    # Detector construction
    g4.detectorConstruction()
    
    # Particle gun setup
    gun = Gun()
    gun.particle = 'gamma'
    gun.energy = '5 keV'
    gun.multiplicity = 1
    g4.addAction(gun)
    
    # Tracker setup
    g4.setOption('tracker', 'MaterialTester')
    
    # Physics list
    g4.setOption('physicsList', 'QGSP_BERT')
    
    # UI macro if provided
    if len(sys.argv) > 1:
        macro_file = sys.argv[1]
        g4.setOption('macro', macro_file)
    
    # Execute Geant4 instance
    g4.run()

if __name__ == '__main__':
    setup_dd4hep_simulation()
```