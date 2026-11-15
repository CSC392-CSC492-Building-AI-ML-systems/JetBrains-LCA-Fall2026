```python
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# Set global simulation parameters
VOLUME_WIDTH_MM = 30
VOLUME_HEIGHT_MM = 30
VOLUME_DEPTH_MM = 10
SPACING_MM = 0.5
WAVELENGTHS = [700, 800, 900]  # nm

# Define simulation path
SIMULATION_PATH = "simpa_tissue_simulation"
os.makedirs(SIMULATION_PATH, exist_ok=True)

# Define global settings
settings = {
    "volume_name": "TissueVolume",
    "volume_dimensions_mm": [VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, VOLUME_DEPTH_MM],
    "spacing_mm": SPACING_MM,
    "wavelengths": WAVELENGTHS,
    "output_path": SIMULATION_PATH,
    "device": sp.devices.ImalacDevice(),
    "number_of_photons": 1e6,
    "do_initial_pressure": True,
    "do_reconstruction": True,
    "reconstruction_module": sp.modules.ReconstructionModuleType.FFT,
    "reconstruction_settings": {
        "speed_of_sound": 1540
    }
}

# Function to create tissue structure
def create_tissue_structure(volume_dimensions_mm, spacing_mm, wavelengths):
    background = {
        "name": "background",
        "structure_type": sp.StructureType.BACKGROUND,
        "background_mua": 0.01,
        "background_mus": 1.0,
        "background_anfact": 0.9,
        "background_g": 0.9,
        "background_ri": 1.0
    }
    muscle_layer = {
        "name": "muscle",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 2,
        "z_end_mm": 8,
        "muscle_mua": 0.05,
        "muscle_mus": 1.5,
        "muscle_anfact": 0.9,
        "muscle_g": 0.9,
        "muscle_ri": 1.37
    }
    epidermis_layer = {
        "name": "epidermis",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 0,
        "z_end_mm": 0.5,
        "epidermis_mua": 0.2,
        "epidermis_mus": 2.0,
        "epidermis_anfact": 0.9,
        "epidermis_g": 0.9,
        "epidermis_ri": 1.4
    }
    blood_vessel_1 = {
        "name": "blood_vessel_1",
        "structure_type": sp.StructureType.CYLINDER,
        "start_mm": [10, 15, 3],
        "end_mm": [10, 15, 7],
        "radius_mm": 1.0,
        "blood_mua": 0.5,
        "blood_mus": 1.0,
        "blood_anfact": 0.9,
        "blood_g": 0.9,
        "blood_ri": 1.4
    }
    blood_vessel_2 = {
        "name": "blood_vessel_2",
        "structure_type": sp.StructureType.CYLINDER,
        "start_mm": [20, 10, 4],
        "end_mm": [20, 10, 8],
        "radius_mm": 1.2,
        "blood_mua": 0.5,
        "blood_mus": 1.0,
        "blood_anfact": 0.9,
        "blood_g": 0.9,
        "blood_ri": 1.4
    }
    tissue_dict = {
        "background": background,
        "muscle": muscle_layer,
        "epidermis": epidermis_layer,
        "blood_vessel_1": blood_vessel_1,
        "blood_vessel_2": blood_vessel_2
    }
    return tissue_dict

# Create tissue structure
tissue_structure = create_tissue_structure(
    [VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, VOLUME_DEPTH_MM],
    SPACING_MM,
    WAVELENGTHS
)

# Create simulation pipeline
pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings, tissue_structure),
    sp.MonteCarloLightTransportAdapter(settings),
    sp.InitialPressureAdapter(settings),
    sp.ReconstructionAdapter(settings)
]

# Run simulation
sp.simulate(pipeline, settings)

# Linear unmixing
unmixing = sp.modules.linear_unmixing.LinearUnmixingModule(settings)
unmixing.run()

# Load and visualize results
for wavelength in WAVELENGTHS:
    pressure_file = os.path.join(SIMULATION_PATH, f"p0_{wavelength}.npy")
    if os.path.exists(pressure_file):
        pressure = np.load(pressure_file)
        plt.figure()
        plt.imshow(pressure[:, :, pressure.shape[2] // 2], cmap='hot')
        plt.title(f"Initial Pressure at {wavelength} nm")
        plt.colorbar()
        plt.show()

# Visualize linear unmixing results
unmixing_file = os.path.join(SIMULATION_PATH, "unmixing_result.npy")
if os.path.exists(unmixing_file):
    unmixing_result = np.load(unmixing_file)
    plt.figure()
    plt.imshow(unmixing_result[:, :, unmixing_result.shape[2] // 2, 0], cmap='hot')
    plt.title("Linear Unmixing Result (First Chromophore)")
    plt.colorbar()
    plt.show()
```