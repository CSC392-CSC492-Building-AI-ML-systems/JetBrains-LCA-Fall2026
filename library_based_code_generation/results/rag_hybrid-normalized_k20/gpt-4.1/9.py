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
        setDumpDetailedParticleInfo,
        loadDDG4,
        import_geometry,
        get_code,
        _configureDD4HEP,
        commandsPreRun,
        compileAClick,
        _configureEDM4HEP,
        setupGun,
        run,
        setPrintLevel,
        setupUserPhysics,
        makeSet,
        addParametersToRunHeader,
        setupPhysics,
        forceDD4HEP,
        _set
    )

    parser = argparse.ArgumentParser(description="DD4hep Simulation Runner")
    parser.add_argument('--geometry', type=str, required=True, help='Path to the geometry file (XML)')
    parser.add_argument('--events', type=int, default=10, help='Number of events to simulate')
    parser.add_argument('--print-level', type=int, default=1, help='Print level for simulation output')
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    args = parser.parse_args()

    logger.info("Loading geometry from %s", args.geometry)
    detector = import_geometry(args.geometry)
    if detector is None:
        logger.error("Failed to load geometry file: %s", args.geometry)
        sys.exit(1)

    logger.info("Loading constants and configuring DD4hep")
    _configureDD4HEP(detector)
    loadDDG4(detector)
    enableDetailedHitsAndParticleInfo(detector)
    setDumpDetailedParticleInfo(detector, True)

    logger.info("Setting up Geant4 simulation")
    sim = DD4hepSimulation(detector)
    setPrintLevel(sim, args.print_level)

    logger.info("Printing detectors")
    for det in detector.detectors():
        logger.info("Detector: %s", det)

    logger.info("Configuring UI, tracking field, and event actions")
    commandsPreRun(sim)
    sim.setupUI()
    sim.setupTrackingField()
    sim.setupEventActions()

    logger.info("Setting up particle gun and tracker")
    setupGun(sim)
    sim.setupTracker()

    logger.info("Building physics list")
    setupPhysics(sim)
    setupUserPhysics(sim)

    logger.info("Executing Geant4 simulation")
    run(sim, args.events)

if __name__ == "__main__":
    run_simulation()
```