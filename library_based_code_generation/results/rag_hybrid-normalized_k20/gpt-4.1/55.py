```python
import sys
import argparse
import logging
from dd4hep.simulation import DD4hepSimulation, enableDetailedHitsAndParticleInfo, get_code
from dd4hep.simulation import setupUserPhysics, commandsInitialize, commandsPreRun, commandsPostRun
from dd4hep.simulation import GeneratorAction, setupUserParticleHandler, setupPhysics, addParametersToRunHeader
from dd4hep.simulation import particlesPerEvent, _configureDD4HEP, PhysicsList, addPhysics, _configureEDM4HEP
from dd4hep.simulation import run, setupDetectors, setupDetector
from dd4hep import Geant4

def display_help():
    help_text = """
DD4hep Simulation Help

Usage:
    python simulation.py [options]

Options:
    -h, --help              Show this help message and exit
    -g, --geometry FILE     Specify the detector geometry XML file
    -n, --nEvents N         Number of events to simulate (default: 10)
    -v, --visualize         Enable visualization
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
    --seed SEED             Set random seed
    --output FILE           Output file name
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-g', '--geometry', type=str, required=True)
    parser.add_argument('-n', '--nEvents', type=int, default=10)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default='output.root')
    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    # Setup logger
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger("DD4hepSimulation")
    logger.info("Starting DD4hep simulation")

    # Setup DD4hepSimulation
    sim = DD4hepSimulation()
    sim.output_file = args.output

    # Parse geometry
    logger.info(f"Loading geometry: {args.geometry}")
    sim.loadGeometry(args.geometry)

    # Configure DD4HEP
    _configureDD4HEP(sim)

    # Setup Geant4 kernel
    logger.info("Setting up Geant4 kernel")
    sim.kernel = Geant4.Kernel()
    sim.detector = sim.detectorDescription()

    # Setup detectors
    logger.info("Setting up detectors")
    setupDetectors(sim)
    setupDetector(sim)

    # Enable detailed hits and particle info
    enableDetailedHitsAndParticleInfo(sim)

    # Setup magnetic field tracking
    logger.info("Configuring magnetic field tracking")
    sim.setupMagneticField()

    # Setup random generator
    if args.seed is not None:
        logger.info(f"Setting random seed: {args.seed}")
        sim.setRandomSeed(args.seed)

    # Setup event actions
    logger.info("Setting up event actions")
    sim.setupEventActions()

    # Setup I/O
    logger.info(f"Configuring output: {args.output}")
    sim.setupIO(args.output)

    # Setup generator actions
    logger.info("Setting up generator actions")
    sim.generator = GeneratorAction()
    sim.addGeneratorAction(sim.generator)

    # Handle simulation particles
    logger.info("Setting up user particle handler")
    setupUserParticleHandler(sim)

    # Build physics list
    logger.info("Building physics list")
    physics_list = PhysicsList()
    addPhysics(physics_list)
    sim.physicsList = physics_list
    setupPhysics(sim)

    # Add special particle types and global range cut
    logger.info("Adding special particle types and global range cut")
    sim.addSpecialParticles()
    sim.setGlobalRangeCut(1.0)  # 1 mm

    # Add parameters to run header
    addParametersToRunHeader(sim)

    # Configure EDM4HEP
    _configureEDM4HEP(sim)

    # Setup user physics
    setupUserPhysics(sim)

    # Setup user interface
    logger.info("Setting up user interface")
    sim.setupUserInterface()

    # Visualization
    if args.visualize:
        logger.info("Enabling visualization")
        sim.enableVisualization()
        sim.executeCommand("/vis/open OGL")
        sim.executeCommand("/vis/drawVolume")
        sim.executeCommand("/vis/viewer/set/viewpointThetaPhi 90 0")
        sim.executeCommand("/vis/scene/add/axes 0 0 0 1.0")
        sim.executeCommand("/vis/scene/add/eventID")
        sim.executeCommand("/vis/scene/add/trajectories smooth")
        sim.executeCommand("/vis/scene/endOfEventAction accumulate")

    # Initialize commands
    commandsInitialize(sim)

    # Pre-run commands
    commandsPreRun(sim)

    # Configure, initialize, and run kernel
    logger.info("Configuring and initializing kernel")
    sim.configure()
    sim.initialize()
    logger.info(f"Running simulation for {args.nEvents} events")
    run(sim, args.nEvents)

    # Post-run commands
    commandsPostRun(sim)

    # Terminate kernel
    logger.info("Terminating kernel")
    sim.terminate()

if __name__ == "__main__":
    main()
```