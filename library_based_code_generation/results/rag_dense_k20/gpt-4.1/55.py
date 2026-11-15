```python
import sys
import argparse
import logging
from dd4hep.simulation import DD4hepSimulation, _configureDD4HEP, setupEDM4hepOutput, _configureEDM4HEP
from dd4hep.simulation import forceDD4HEP, loaddd4hep, loadDDG4, Geant4, setupUserPhysics, setDumpDetailedParticleInfo
from dd4hep.simulation import useHepMC3, setupPhysics, setupCalorimeter, enableDetailedHitsAndParticleInfo
from dd4hep.simulation import setupUserParticleHandler, forceEDM4HEP, pdgfile, commandsInitialize

def display_help():
    help_text = """
DD4hep Simulation Help

Usage:
    python simulation.py [options]

Options:
    -h, --help              Show this help message and exit
    -g, --geometry FILE     Path to the detector geometry XML file
    -n, --nevents N         Number of events to simulate (default: 10)
    -v, --visualize         Enable visualization
    -o, --output FILE       Output file name (default: output.edm4hep.root)
    --random-seed SEED      Set random seed
    --range-cut VALUE       Set global range cut (in mm)
    --log-level LEVEL       Set logging level (DEBUG, INFO, WARNING, ERROR)
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='store_true')
    parser.add_argument('-g', '--geometry', type=str, required=True)
    parser.add_argument('-n', '--nevents', type=int, default=10)
    parser.add_argument('-v', '--visualize', action='store_true')
    parser.add_argument('-o', '--output', type=str, default='output.edm4hep.root')
    parser.add_argument('--random-seed', type=int, default=None)
    parser.add_argument('--range-cut', type=float, default=0.1)
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()

    if args.help:
        display_help()
        sys.exit(0)

    # Setup logger
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='[%(levelname)s] %(message)s')

    logging.info("Starting DD4hep simulation setup...")

    # Load geometry
    logging.info(f"Loading geometry from {args.geometry}")
    detector = loaddd4hep(args.geometry)

    # Configure DD4hep simulation
    sim = DD4hepSimulation(detector)
    _configureDD4HEP(sim)

    # Setup Geant4 kernel
    kernel = Geant4.Kernel()
    sim.kernel = kernel

    # Setup user interface
    sim.setupUI()

    # Magnetic field tracking
    sim.setupMagneticField()

    # Random generator
    if args.random_seed is not None:
        logging.info(f"Setting random seed: {args.random_seed}")
        sim.setRandomSeed(args.random_seed)

    # Event actions
    sim.setupEventActions()

    # I/O
    logging.info(f"Setting up EDM4hep output: {args.output}")
    setupEDM4hepOutput(sim, args.output)
    _configureEDM4HEP(sim)

    # Generator actions
    useHepMC3(sim)
    sim.setupPrimaryGenerator()
    sim.setupGeneratorActions()

    # Simulation particles
    sim.setupSimulationParticles()

    # Setup detectors
    setupCalorimeter(sim)
    sim.setupSensitiveDetectors()

    # Build physics list
    setupPhysics(sim)
    setupUserPhysics(sim)

    # Add special particle types
    setupUserParticleHandler(sim)

    # Global range cut
    logging.info(f"Setting global range cut: {args.range_cut} mm")
    sim.setGlobalRangeCut(args.range_cut)

    # Enable detailed hits and particle info
    enableDetailedHitsAndParticleInfo(sim)
    setDumpDetailedParticleInfo(sim, True)

    # Visualization
    if args.visualize:
        logging.info("Visualization enabled.")
        sim.enableVisualization()
        sim.addVisualizationCommands([
            "/vis/open OGL",
            "/vis/drawVolume",
            "/vis/viewer/set/viewpointThetaPhi 90 0",
            "/vis/scene/add/trajectories smooth",
            "/vis/scene/endOfEventAction accumulate"
        ])

    # Configure, initialize, run, and terminate kernel
    logging.info("Configuring kernel...")
    sim.configure()
    logging.info("Initializing kernel...")
    sim.initialize()
    logging.info(f"Running simulation for {args.nevents} events...")
    sim.run(args.nevents)
    logging.info("Terminating kernel...")
    sim.terminate()
    logging.info("Simulation completed.")

if __name__ == "__main__":
    main()
```