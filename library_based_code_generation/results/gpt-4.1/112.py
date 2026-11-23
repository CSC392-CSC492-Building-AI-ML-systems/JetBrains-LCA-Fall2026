```python
import logging
import dd4hep
from dd4hep import Detector
from dd4hep.simulation import Kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dd4hep_simulation")

def run_dd4hep_simulation():
    import dd4hep
    from dd4hep.simulation import Kernel, Geant4, EventAction, ParticleGun, PhysicsList
    from dd4hep.core import constants as dd4hep_constants

    logger.info("Initializing DD4hep kernel...")
    kernel = Kernel()
    detector = Detector.getInstance()
    
    geometry_file = "detector.xml"
    logger.info(f"Loading geometry from {geometry_file}...")
    detector.fromXML(geometry_file)
    
    logger.info("Importing constants...")
    detector.apply("file:constants.xml")
    
    logger.info("Configuring Geant4 interface...")
    geant4 = Geant4(kernel)
    
    logger.info("Setting up tracking field...")
    detector.setTrackingField("MagneticField")
    
    logger.info("Configuring event actions...")
    event_action = EventAction()
    geant4.eventAction = event_action
    
    logger.info("Setting up particle gun...")
    particle_gun = ParticleGun()
    particle_gun.particleName = "e-"
    particle_gun.energy = 10 * dd4hep_constants.GeV
    particle_gun.direction = (0, 0, 1)
    geant4.generator = particle_gun
    
    logger.info("Handling simulation particles...")
    # Example: Add more particles or configure as needed
    
    logger.info("Building physics list...")
    physics_list = PhysicsList("FTFP_BERT")
    geant4.physicsList = physics_list
    
    logger.info("Starting simulation engine...")
    kernel.run()

if __name__ == "__main__":
    run_dd4hep_simulation()
```