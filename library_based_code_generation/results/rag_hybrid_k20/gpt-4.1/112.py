```python
import logging
import dd4hep
from dd4hep.simulation import DD4hepSimulation, _configureDD4HEP, enableDetailedHitsAndParticleInfo, import_geometry, get_code, setDumpDetailedParticleInfo
from dd4hep.simulation import forceDD4HEP, _configureEDM4HEP, commandsPreRun, setupGun, setupPhysics, loadDDG4, addParametersToRunHeader, particlesPerEvent, import_tgeo, run, setupUserPhysics, commandsPostRun, _import_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_dd4hep_simulation():
    import dd4hep
    import dd4hep.simulation
    import dd4hep.geant4
    from dd4hep.simulation import Geant4

    logger.info("Configuring DD4hep simulation kernel...")
    kernel = dd4hep.simulation.DD4hepSimulation()
    _configureDD4HEP(kernel)
    forceDD4HEP(kernel)

    geometry_file = "detector.xml"
    logger.info(f"Importing geometry from {geometry_file}...")
    import_geometry(kernel, geometry_file)
    import_tgeo(kernel, geometry_file)

    logger.info("Importing constants...")
    _import_module("my_constants")  # Replace with your constants module if needed

    logger.info("Configuring Geant4 interface...")
    g4 = Geant4(kernel)
    loadDDG4(g4)
    setupUserPhysics(g4)
    setupPhysics(g4)
    _configureEDM4HEP(g4)

    logger.info("Setting up tracking field and event actions...")
    enableDetailedHitsAndParticleInfo(g4)
    setDumpDetailedParticleInfo(g4, True)
    addParametersToRunHeader(g4)

    logger.info("Setting up particle gun...")
    setupGun(g4)
    particlesPerEvent(g4, 1)

    logger.info("Handling simulation particles...")
    commandsPreRun(g4)
    commandsPostRun(g4)

    logger.info("Building physics list...")
    # Physics list is set up in setupPhysics and setupUserPhysics

    logger.info("Starting simulation engine...")
    run(g4)

if __name__ == "__main__":
    run_dd4hep_simulation()
```