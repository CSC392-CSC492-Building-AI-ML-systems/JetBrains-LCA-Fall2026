```python
import sys
import argparse
import logging
from dd4hep.simulation import DD4hepSimulation, enableDetailedHitsAndParticleInfo, _configureDD4HEP, get_code, setupUserPhysics, commandsInitialize, commandsPreRun, Geant4, GeneratorAction, setupUserParticleHandler, setupPhysics, addParametersToRunHeader, _configureEDM4HEP, commandsPostRun, forceDD4HEP, particlesPerEvent, PhysicsList, addPhysics, run, setupDetectors

def display_help():
    help_text = """
    DD4hep Simulation Help

    Usage:
      python simulation.py [options]

    Options:
      -h, --help            Show this help message and exit
      -g, --geometry FILE   Specify the geometry XML file to load
      -v, --visualize       Enable visualization
      -n, --nEvents N       Number of events to simulate
      --seed SEED           Set random seed
      --log-level LEVEL     Set logging level (DEBUG, INFO, WARNING, ERROR)
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-g', '--geometry', type=str, required=True, help='Geometry XML file')
    parser.add_argument('-v', '--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('-n', '--nEvents', type=int, default=10, help='Number of events')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level')
    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    # Setup logger
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("DD4hepSimulation")
    logger.info("Starting DD4hep simulation setup...")

    # Setup DD4hep simulation kernel
    sim = DD4hepSimulation()
    enableDetailedHitsAndParticleInfo(sim)
    _configureDD4HEP(sim)
    forceDD4HEP(sim)

    # Load geometry
    logger.info(f"Loading geometry from {args.geometry}")
    sim.loadGeometry(args.geometry)

    # Setup detectors
    setupDetectors(sim)

    # Setup magnetic field tracking
    sim.setupMagneticField()

    # Setup random generator
    sim.setRandomSeed(args.seed)

    # Setup simulation particles
    setupUserParticleHandler(sim)

    # Build physics list and add special particle types
    physics = PhysicsList()
    addPhysics(physics, "FTFP_BERT")
    addPhysics(physics, "SpecialParticles")
    sim.setPhysicsList(physics)
    setupPhysics(sim)

    # Add global range cut
    sim.setGlobalRangeCut(0.1)  # in mm

    # Setup event actions
    sim.setupEventActions()

    # Setup I/O
    _configureEDM4HEP(sim)
    sim.setupIO("output.root")

    # Setup generator actions
    gen_action = GeneratorAction()
    sim.addGeneratorAction(gen_action)
    particlesPerEvent(sim, 1)

    # Add parameters to run header
    addParametersToRunHeader(sim, {"geometry": args.geometry, "nEvents": args.nEvents})

    # Configure user interface
    sim.setupUserInterface()

    # Visualization commands
    if args.visualize:
        logger.info("Visualization enabled.")
        sim.addCommand("/vis/open OGL")
        sim.addCommand("/vis/drawVolume")
        sim.addCommand("/vis/viewer/set/viewpointThetaPhi 90 0")
        sim.addCommand("/vis/scene/add/axes 0 0 0 1.0")
        sim.addCommand("/vis/scene/add/trajectories smooth")
        sim.addCommand("/vis/scene/endOfEventAction accumulate")
        sim.addCommand("/vis/enable")

    # Initialize kernel
    commandsInitialize(sim)

    # Pre-run commands
    commandsPreRun(sim)

    # Run simulation
    logger.info(f"Running simulation for {args.nEvents} events...")
    run(sim, args.nEvents)

    # Post-run commands
    commandsPostRun(sim)

    # Terminate kernel
    sim.terminate()
    logger.info("Simulation finished.")

if __name__ == "__main__":
    main()
```