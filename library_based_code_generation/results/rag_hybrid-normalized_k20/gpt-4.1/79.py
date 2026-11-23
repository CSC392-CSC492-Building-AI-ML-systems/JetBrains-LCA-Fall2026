```python
import os
import sys
from DDG4 import DD4hepSimulation, Geant4

def setup_dd4hep_simulation():
    # Initialize DDG4 kernel
    sim = DD4hepSimulation()
    kernel = sim.kernel

    # Load geometry from XML file
    install_dir = os.environ.get('DD4hepExamplesINSTALL')
    if not install_dir:
        raise EnvironmentError("DD4hepExamplesINSTALL environment variable not set.")
    geometry_path = os.path.join(install_dir, 'OpticalSurfaces', 'compact', 'OpticalSurfaces.xml')
    kernel.loadGeometry(geometry_path)

    # Import constants from detector description
    kernel.detectorDescription().importConstants(kernel)

    # Configure Geant4 instance
    g4 = Geant4(kernel)
    g4.setupUI()

    # Use macro if provided as command line argument
    if len(sys.argv) > 1:
        macro_file = sys.argv[1]
        g4.setUIMacro(macro_file)

    # Setup detector construction
    g4.setupDetector()

    # Setup physics list
    g4.setupPhysics('QGSP_BERT')

    # Setup tracker
    g4.addTracker('MaterialTester')

    # Setup particle gun
    gun = g4.setupGun()
    gun.particle = 'gamma'
    gun.energy = 5e-3  # 5 keV in GeV
    gun.multiplicity = 1

    # Setup event actions (if needed)
    g4.setupEventActions()

    # Setup tracking field (if needed)
    g4.setupTrackingField()

    # Execute Geant4 instance
    g4.execute()

if __name__ == '__main__':
    setup_dd4hep_simulation()
```