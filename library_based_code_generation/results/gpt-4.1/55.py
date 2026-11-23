```python
import sys
import argparse
import logging
from dd4hep import DD4hep
from dd4hep.simulation import Kernel, Geant4Kernel, DetectorConstruction, PhysicsList, EventAction, GeneratorAction, IOManager
from dd4hep.simulation import ParticleGun, ParticleHandler, MagneticField, RandomGenerator, RangeCut
from dd4hep.simulation import VisualizationManager

def display_help():
    help_text = """
    DD4hep Simulation Script

    Usage:
      python simulate.py [options]

    Options:
      -h, --help            Show this help message and exit
      -g, --geometry FILE   Path to the detector geometry XML file
      -n, --nEvents N       Number of events to simulate (default: 10)
      -v, --visualize       Enable visualization
      -o, --output FILE     Output file name (default: output.root)
      --seed SEED           Random seed (default: 12345)
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-g', '--geometry', type=str, required=True)
    parser.add_argument('-n', '--nEvents', type=int, default=10)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-o', '--output', type=str, default='output.root')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dd4hep.simulation")
    logger.info("Starting DD4hep simulation")

    # Initialize DD4hep and Geant4 kernel
    dd4hep_instance = DD4hep.DD4hep()
    kernel = Geant4Kernel()

    # Load detector geometry
    logger.info(f"Loading geometry from {args.geometry}")
    detector = DetectorConstruction(args.geometry)
    kernel.setDetectorConstruction(detector)

    # Setup magnetic field
    logger.info("Setting up magnetic field")
    magnetic_field = MagneticField()
    kernel.setField(magnetic_field)

    # Setup random generator
    logger.info(f"Setting random seed: {args.seed}")
    random_gen = RandomGenerator(seed=args.seed)
    kernel.setRandomGenerator(random_gen)

    # Setup event actions
    logger.info("Configuring event actions")
    event_action = EventAction()
    kernel.addEventAction(event_action)

    # Setup I/O
    logger.info(f"Configuring I/O: {args.output}")
    io_manager = IOManager(args.output)
    kernel.setIOManager(io_manager)

    # Setup generator actions
    logger.info("Setting up generator actions")
    particle_gun = ParticleGun()
    generator_action = GeneratorAction(particle_gun)
    kernel.addGeneratorAction(generator_action)

    # Handle simulation particles
    logger.info("Setting up particle handler")
    particle_handler = ParticleHandler()
    kernel.setParticleHandler(particle_handler)

    # Setup detectors
    logger.info("Setting up detectors")
    detector.setupDetectors(kernel)

    # Build physics list
    logger.info("Building physics list")
    physics_list = PhysicsList("FTFP_BERT")
    kernel.setPhysicsList(physics_list)

    # Add special particle types
    logger.info("Adding special particle types")
    physics_list.addSpecialParticles(["opticalphoton", "geantinos"])

    # Set global range cut
    logger.info("Setting global range cut")
    range_cut = RangeCut(0.1)  # in mm
    physics_list.setRangeCut(range_cut)

    # Visualization
    if args.visualize:
        logger.info("Enabling visualization")
        vis_manager = VisualizationManager()
        kernel.setVisualizationManager(vis_manager)
        vis_manager.addCommand("/vis/open OGL")
        vis_manager.addCommand("/vis/drawVolume")
        vis_manager.addCommand("/vis/viewer/set/viewpointThetaPhi 90 0")
        vis_manager.addCommand("/vis/scene/add/axes 0 0 0 1")

    # Configure, initialize, run, and terminate kernel
    logger.info("Configuring kernel")
    kernel.configure()
    logger.info("Initializing kernel")
    kernel.initialize()
    logger.info(f"Running simulation for {args.nEvents} events")
    kernel.run(args.nEvents)
    logger.info("Terminating kernel")
    kernel.terminate()
    logger.info("Simulation finished")

if __name__ == "__main__":
    main()
```