```python
import sys
import logging
import dd4hep

def run_simulation():
    import argparse
    from dd4hep import Detector
    from dd4hep.simulation import geant4
    from dd4hep.simulation import setupUI, setupTrackingField, setupEventActions
    from dd4hep.simulation import ParticleGun, Tracker
    from dd4hep.simulation import PhysicsList

    # Set up command line arguments
    parser = argparse.ArgumentParser(description="DD4hep Geant4 Simulation")
    parser.add_argument('-g', '--geometry', type=str, required=True, help='Path to the geometry XML file')
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("dd4hep.simulation")

    # Load geometry
    logger.info(f"Loading geometry from {args.geometry}")
    detector = Detector.getInstance()
    detector.fromXML(args.geometry)

    # Load constants
    logger.info("Loading constants")
    detector.apply("DD4hep_Constants")

    # Set up Geant4
    logger.info("Setting up Geant4")
    g4 = geant4.G4Simulation(detector)

    # Print detectors
    logger.info("Available detectors:")
    for det in detector.detectors():
        logger.info(f" - {det.name}")

    # Configure UI
    logger.info("Configuring Geant4 UI")
    setupUI(g4)

    # Configure tracking field
    logger.info("Configuring tracking field")
    setupTrackingField(g4, detector)

    # Configure event actions
    logger.info("Configuring event actions")
    setupEventActions(g4)

    # Set up particle gun
    logger.info("Setting up particle gun")
    gun = ParticleGun()
    g4.setGenerator(gun)

    # Set up tracker
    logger.info("Setting up tracker")
    tracker = Tracker()
    g4.addSensitiveDetector(tracker)

    # Build physics list
    logger.info("Building physics list")
    physics = PhysicsList("FTFP_BERT")
    g4.setPhysicsList(physics)

    # Execute Geant4
    logger.info("Executing Geant4 simulation")
    g4.run()

if __name__ == "__main__":
    run_simulation()
```