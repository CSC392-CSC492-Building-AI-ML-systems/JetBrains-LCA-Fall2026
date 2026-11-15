```python
import logging
from dd4hep.simulation import DD4hepSimulation, setOptions, setOption, run
from dd4hep.simulation import import_geometry, importConstants, enableDetailedHitsAndParticleInfo
from dd4hep.simulation import _setKernelProperty, makeSet
from dd4hep.simulation import import_units

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_dd4hep_simulation():
    # Import additional modules
    from dd4hep.simulation import import_root, import_detail, import_rec, import_tgeo

    # Set up the simulation kernel
    sim = DD4hepSimulation()
    logger.info("DD4hepSimulation kernel created.")

    # Load geometry from file
    geometry_file = "detector.xml"
    import_geometry(sim, geometry_file)
    logger.info(f"Geometry loaded from {geometry_file}.")

    # Import constants
    importConstants(sim, "constants.xml")
    logger.info("Constants imported.")

    # Enable detailed hits and particle info
    enableDetailedHitsAndParticleInfo(sim)
    logger.info("Detailed hits and particle info enabled.")

    # Configure Geant4 interface
    setOption(sim, "Simulation", "Geant4")
    logger.info("Geant4 interface configured.")

    # Set up the tracking field
    setOption(sim, "Field", "UniformMagneticField")
    setOption(sim, "FieldStrength", "0.0 0.0 2.0")  # Example: 2T along z
    logger.info("Tracking field set.")

    # Set up event actions
    setOption(sim, "EventAction", "DefaultEventAction")
    logger.info("Event actions set.")

    # Set up the particle gun
    setOption(sim, "Generator", "ParticleGun")
    setOption(sim, "ParticleGun.Particle", "e-")
    setOption(sim, "ParticleGun.Momentum", "10*GeV")
    setOption(sim, "ParticleGun.Theta", "90*deg")
    setOption(sim, "ParticleGun.Phi", "0*deg")
    logger.info("Particle gun configured.")

    # Handle simulation particles
    setOption(sim, "Particles", "all")
    logger.info("Simulation particles set.")

    # Build the physics list
    setOption(sim, "PhysicsList", "FTFP_BERT")
    logger.info("Physics list built.")

    # Start the simulation engine
    run(sim)
    logger.info("Simulation started.")

if __name__ == "__main__":
    run_dd4hep_simulation()
```