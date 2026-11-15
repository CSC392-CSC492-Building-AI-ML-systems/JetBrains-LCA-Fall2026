```python
import sys
import logging
import argparse
from dd4hep.simulation import DD4hepSimulation, GeneratorAction, MagneticField, kernel, run, _kernel_terminate
from dd4hep.simulation import setupTrackingField, setupUserPhysics, enableDetailedHitsAndParticleInfo
from dd4hep.simulation import addParametersToRunHeader, mapActions, commandsPreRun, commandsPostRun, commandsInitialize, commandsTerminate
from dd4hep.simulation import _configureDD4HEP

def display_help():
    help_text = """
DD4hep Simulation Help

Usage:
    python simulation.py [options]

Options:
    -h, --help              Show this help message and exit
    --geometry FILE         Path to the detector geometry XML file
    --events N              Number of events to simulate
    --vis                   Enable visualization
    --random-seed SEED      Set random seed
    --output FILE           Output file name
    --magnetic-field FILE   Magnetic field configuration file
    --physics-list NAME     Physics list to use
    --range-cut VALUE       Global range cut value (in mm)
    --detailed-hits         Enable detailed hits and particle info
    """
    print(help_text)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('--geometry', type=str, required=True)
    parser.add_argument('--events', type=int, default=10)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--output', type=str, default='output.root')
    parser.add_argument('--magnetic-field', type=str, default=None)
    parser.add_argument('--physics-list', type=str, default='FTFP_BERT')
    parser.add_argument('--range-cut', type=float, default=0.1)
    parser.add_argument('--detailed-hits', action='store_true')
    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DD4hepSimulation")
    logger.info("Starting DD4hep simulation setup...")

    # Configure DD4hep kernel and detector description
    sim = DD4hepSimulation()
    _configureDD4HEP(sim, args.geometry)

    # Setup magnetic field tracking if specified
    if args.magnetic_field:
        logger.info(f"Loading magnetic field from {args.magnetic_field}")
        MagneticField(sim, args.magnetic_field)
        setupTrackingField(sim)

    # Setup random generator
    if args.random_seed is not None:
        logger.info(f"Setting random seed: {args.random_seed}")
        sim.random.setSeed(args.random_seed)

    # Enable detailed hits and particle info if requested
    if args.detailed_hits:
        enableDetailedHitsAndParticleInfo(sim)

    # Setup event actions, I/O, and generator actions
    logger.info("Setting up event actions and I/O")
    addParametersToRunHeader(sim)
    sim.output.fileName = args.output

    # Setup generator actions
    logger.info("Setting up generator actions")
    sim.generator = GeneratorAction(sim)
    mapActions(sim)

    # Handle simulation particles and setup detectors
    logger.info("Setting up detectors")
    sim.setupDetectors()

    # Build physics list
    logger.info(f"Building physics list: {args.physics_list}")
    setupUserPhysics(sim, args.physics_list)

    # Add special particle types and global range cut
    logger.info(f"Setting global range cut: {args.range_cut} mm")
    sim.physicsList.setRangeCut(args.range_cut)

    # Visualization commands if enabled
    if args.vis:
        logger.info("Enabling visualization")
        sim.commands.append("/vis/open OGL")
        sim.commands.append("/vis/drawVolume")
        sim.commands.append("/vis/viewer/set/viewpointThetaPhi 30 30")
        sim.commands.append("/vis/scene/add/trajectories smooth")
        sim.commands.append("/vis/scene/endOfEventAction accumulate")

    # Configure, initialize, run, and terminate the kernel
    logger.info("Configuring kernel")
    commandsInitialize(sim)
    logger.info("Running simulation")
    commandsPreRun(sim)
    run(sim, args.events)
    commandsPostRun(sim)
    commandsTerminate(sim)
    _kernel_terminate(sim)
    logger.info("Simulation finished.")

if __name__ == "__main__":
    main()
```