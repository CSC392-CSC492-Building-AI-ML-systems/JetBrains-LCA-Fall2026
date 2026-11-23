```python
import logging
from DD4hepSimulation import DD4hepSimulation, enableDetailedHitsAndParticleInfo, setDumpDetailedParticleInfo
from DD4hepSimulation import import_geometry, get_code, import_tgeo, loadDDG4, _configureDD4HEP, _configureEDM4HEP
from DD4hepSimulation import addParametersToRunHeader, particlesPerEvent, setupPhysics, setupGun, setupUserPhysics
from DD4hepSimulation import forceDD4HEP, commandsPreRun, commandsPostRun, run, _import_module

def run_dd4hep_simulation():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DD4hepSimulation")
    logger.info("Starting DD4hep simulation setup.")

    # Import additional modules
    import Geant4

    # Set up the DD4hep kernel
    kernel = DD4hepSimulation()
    logger.info("DD4hep kernel initialized.")

    # Load geometry from file
    geometry_file = "detector.xml"
    import_geometry(kernel, geometry_file)
    logger.info(f"Geometry loaded from {geometry_file}.")

    # Import constants
    constants_module = _import_module("detector_constants")
    logger.info("Detector constants imported.")

    # Configure Geant4 interface
    loadDDG4(kernel)
    logger.info("Geant4 interface configured.")

    # Set up tracking field
    kernel.field = getattr(constants_module, "tracking_field", None)
    logger.info("Tracking field set.")

    # Set up event actions
    enableDetailedHitsAndParticleInfo(kernel)
    setDumpDetailedParticleInfo(kernel, True)
    logger.info("Event actions configured.")

    # Set up particle gun
    setupGun(kernel, particlesPerEvent=1)
    logger.info("Particle gun configured.")

    # Handle simulation particles
    forceDD4HEP(kernel)
    logger.info("Simulation particles handled.")

    # Build physics list
    setupPhysics(kernel, physics_list="FTFP_BERT")
    setupUserPhysics(kernel)
    logger.info("Physics list built.")

    # Configure EDM4HEP
    _configureEDM4HEP(kernel)
    logger.info("EDM4HEP configured.")

    # Add parameters to run header
    addParametersToRunHeader(kernel)
    logger.info("Run header parameters added.")

    # Pre-run commands
    commandsPreRun(kernel)
    logger.info("Pre-run commands executed.")

    # Start the simulation engine
    run(kernel)
    logger.info("Simulation engine started.")

    # Post-run commands
    commandsPostRun(kernel)
    logger.info("Post-run commands executed.")

if __name__ == "__main__":
    run_dd4hep_simulation()
```