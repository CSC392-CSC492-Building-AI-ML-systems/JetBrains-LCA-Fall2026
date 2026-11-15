```python
import sys
import logging
from dd4hep.simulation import DD4hepSimulation, _configureDD4HEP, setupEDM4hepOutput, setupGun, setupPhysics
from dd4hep.simulation import setupUserPhysics, setupUserParticleHandler, loadDDG4, loaddd4hep
from dd4hep.simulation import commandsPreRun, commandsInitialize, setDumpDetailedParticleInfo, enableDetailedHitsAndParticleInfo
from dd4hep.simulation import Geant4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_simulation():
    import argparse

    parser = argparse.ArgumentParser(description="DD4hep Simulation Runner")
    parser.add_argument('-g', '--geometry', type=str, required=True, help='Path to the geometry file (XML)')
    parser.add_argument('-n', '--nEvents', type=int, default=10, help='Number of events to simulate')
    parser.add_argument('--output', type=str, default='output.edm4hep.root', help='Output file name')
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    args = parser.parse_args()

    logger.info("Loading geometry from: %s", args.geometry)
    detector = loaddd4hep(args.geometry)
    if detector is None:
        logger.error("Failed to load geometry file: %s", args.geometry)
        sys.exit(1)

    logger.info("Loading constants and configuring DD4hep")
    _configureDD4HEP(detector)

    logger.info("Setting up Geant4 simulation")
    sim = DD4hepSimulation(detector)
    setupEDM4hepOutput(sim, args.output)
    setupUserPhysics(sim)
    setupUserParticleHandler(sim)
    setupGun(sim)
    setupPhysics(sim)
    commandsPreRun(sim)
    setDumpDetailedParticleInfo(sim, True)
    enableDetailedHitsAndParticleInfo(sim, True)
    commandsInitialize(sim)

    logger.info("Printing detectors:")
    for det in detector.detectors():
        logger.info(" - %s", det)

    logger.info("Configuring UI, tracking field, and event actions")
    # Placeholder for UI, tracking field, and event actions configuration
    # (Assume these are handled by the simulation setup above or can be added here)

    logger.info("Setting up particle gun and tracker")
    # Particle gun and tracker are set up above via setupGun and setupPhysics

    logger.info("Building physics list and executing Geant4")
    g4 = Geant4(sim)
    g4.run(args.nEvents)

if __name__ == "__main__":
    run_simulation()
```