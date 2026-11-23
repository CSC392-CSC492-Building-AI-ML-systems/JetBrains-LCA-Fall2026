```python
import sys
import logging
from dd4hep import compileAClick, enableDetailedHitsAndParticleInfo, DD4hepSimulation, _set, get_unbound_function, setPrintLevel, setOptions, makeSet, setOption, Geant4, __setMagneticFieldOptions, importConstants, printDetectors, _setKernelProperty, import_geometry, setDumpDetailedParticleInfo, show_help, is_package, get_code, CommandLine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_simulation():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="DD4hep Simulation Runner")
    parser.add_argument('--geometry', type=str, required=True, help='Path to the geometry file (XML)')
    parser.add_argument('--events', type=int, default=10, help='Number of events to simulate')
    parser.add_argument('--help', action='store_true', help='Show help message and exit')
    args = parser.parse_args()

    if args.help:
        show_help()
        sys.exit(0)

    geometry_file = args.geometry
    if not os.path.exists(geometry_file):
        logger.error(f"Geometry file '{geometry_file}' does not exist.")
        sys.exit(1)

    logger.info("Importing geometry...")
    detector = import_geometry(geometry_file)

    logger.info("Importing constants...")
    importConstants(detector)

    logger.info("Setting up Geant4 simulation...")
    sim = DD4hepSimulation(detector)
    setPrintLevel(sim, logging.INFO)
    setOptions(sim, {"events": args.events})

    logger.info("Printing detectors...")
    printDetectors(detector)

    logger.info("Configuring UI...")
    setOption(sim, "ui", True)

    logger.info("Configuring tracking field...")
    __setMagneticFieldOptions(sim, {"field": "default"})

    logger.info("Configuring event actions...")
    setOption(sim, "eventActions", True)

    logger.info("Setting up particle gun...")
    setOption(sim, "particleGun", {"particle": "e-", "energy": 10.0})

    logger.info("Setting up tracker...")
    setOption(sim, "tracker", True)

    logger.info("Building physics list...")
    setOption(sim, "physicsList", "FTFP_BERT")

    logger.info("Executing Geant4 simulation...")
    Geant4(sim)

if __name__ == "__main__":
    run_simulation()
```