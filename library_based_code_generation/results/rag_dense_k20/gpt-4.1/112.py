```python
import logging
from dd4hep.simulation import DD4hepSimulation, _configureDD4HEP, setupEDM4hepOutput, forceDD4HEP
from dd4hep.simulation import test_module_ddg4, _configureEDM4HEP, loaddd4hep, loadDDG4
from dd4hep.simulation import setupUserPhysics, setupPhysics, Geant4
from dd4hep.simulation import setupUserParticleHandler, setupGun, pdgfile
from dd4hep.simulation import commandsPreRun, setDumpDetailedParticleInfo, forceEDM4HEP
from dd4hep.simulation import useHepMC3, _setup, enableDetailedHitsAndParticleInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_dd4hep_simulation():
    import dd4hep
    import dd4hep.simulation
    import dd4hep.core

    logger.info("Configuring DD4hep kernel...")
    kernel = dd4hep.simulation.DD4hepSimulation()
    _configureDD4HEP(kernel)

    geometry_file = "detector.xml"
    logger.info(f"Loading geometry from {geometry_file}...")
    loaddd4hep(kernel, geometry_file)

    constants_file = "constants.xml"
    logger.info(f"Importing constants from {constants_file}...")
    kernel.importConstants(constants_file)

    logger.info("Configuring Geant4 interface...")
    Geant4(kernel)
    loadDDG4(kernel)

    logger.info("Setting up tracking field...")
    kernel.field = "ConstantField"
    kernel.field_strength = (0.0, 0.0, 2.0)  # Example: 2T along z

    logger.info("Setting up event actions...")
    setupUserParticleHandler(kernel)
    setDumpDetailedParticleInfo(kernel, True)
    enableDetailedHitsAndParticleInfo(kernel, True)

    logger.info("Setting up particle gun...")
    setupGun(kernel, particle="e-", energy=10.0, direction=(0,0,1))

    logger.info("Handling simulation particles...")
    pdgfile(kernel, "pdg_table.txt")

    logger.info("Building physics list...")
    setupPhysics(kernel, physics_list="FTFP_BERT")
    setupUserPhysics(kernel)

    logger.info("Setting up EDM4hep output...")
    setupEDM4hepOutput(kernel, filename="output.edm4hep.root")
    forceEDM4HEP(kernel)

    logger.info("Running simulation engine...")
    kernel.run()

if __name__ == "__main__":
    run_dd4hep_simulation()
```