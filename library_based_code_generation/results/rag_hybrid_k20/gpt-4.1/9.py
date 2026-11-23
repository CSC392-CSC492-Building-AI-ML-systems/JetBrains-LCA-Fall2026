```python
import sys
import logging
from dd4hep.simulation import DD4hepSimulation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_simulation():
    import argparse
    from dd4hep.simulation import (
        Geant4,
        enableDetailedHitsAndParticleInfo,
        _configureDD4HEP,
        setDumpDetailedParticleInfo,
        loadDDG4,
        import_geometry,
        _configureEDM4HEP,
        forceDD4HEP,
        get_code,
        commandsPreRun,
        compileAClick,
        setupGun,
        run,
        setPrintLevel,
        makeSet,
        setupUserPhysics,
        addParametersToRunHeader,
        setupPhysics,
        commandsPostRun
    )

    parser = argparse.ArgumentParser(description="DD4hep Simulation Runner")
    parser.add_argument('--geometry', type=str, required=True, help='Path to the geometry file (XML)')
    parser.add_argument('--events', type=int, default=10, help='Number of events to simulate')
    parser.add_argument('--print-level', type=int, default=1, help='Print level for logging')
    parser.add_argument('--help', action='store_true', help='Show help message and exit')
    args, unknown = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    logger.info("Loading geometry from %s", args.geometry)
    detector = import_geometry(args.geometry)

    logger.info("Loading constants and configuring DD4hep")
    _configureDD4HEP(detector)
    forceDD4HEP(detector)
    setDumpDetailedParticleInfo(True)
    enableDetailedHitsAndParticleInfo(detector)

    logger.info("Setting up Geant4 simulation")
    sim = DD4hepSimulation(detector)
    geant4 = Geant4(sim)
    setPrintLevel(args.print_level)

    logger.info("Printing detectors")
    for det in detector.detectors():
        logger.info("Detector: %s", det)

    logger.info("Configuring UI, tracking field, and event actions")
    commandsPreRun(sim)
    setupUserPhysics(sim)
    setupPhysics(sim)
    addParametersToRunHeader(sim)
    commandsPostRun(sim)

    logger.info("Setting up particle gun and tracker")
    setupGun(sim)
    makeSet(sim, "Tracker")

    logger.info("Building physics list")
    setupUserPhysics(sim)

    logger.info("Executing Geant4 simulation")
    run(sim, args.events)

if __name__ == "__main__":
    run_simulation()
```