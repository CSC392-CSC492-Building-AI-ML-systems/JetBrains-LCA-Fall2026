```python
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# Global simulation parameters
VOLUME_WIDTH_MM = 20
VOLUME_HEIGHT_MM = 10
VOLUME_DEPTH_MM = 20
SPACING_MM = 0.2
WAVELENGTHS = [700, 800, 900]  # nm

# Simulation path
SIMULATION_PATH = "simpa_tissue_simulation"

# Function to create tissue structure
def create_tissue_structure():
    background = {
        "name": "background",
        "structure_type": sp.StructureType.BACKGROUND,
        "mus": 0.01,
        "mua": 0.01,
        "g": 0.9,
        "refractive_index": 1.0,
        "priority": 0
    }
    muscle_layer = {
        "name": "muscle",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 2,
        "z_end_mm": 8,
        "mus": 1.0,
        "mua": 0.05,
        "g": 0.9,
        "refractive_index": 1.37,
        "priority": 1
    }
    epidermis_layer = {
        "name": "epidermis",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 0,
        "z_end_mm": 2,
        "mus": 1.5,
        "mua": 0.1,
        "g": 0.9,
        "refractive_index": 1.4,
        "priority": 2
    }
    blood_vessel_1 = {
        "name": "blood_vessel_1",
        "structure_type": sp.StructureType.CYLINDER,
        "start_mm": [5, 5, 3],
        "end_mm": [5, 15, 3],
        "radius_mm": 1,
        "mus": 1.2,
        "mua": 0.2,
        "g": 0.9,
        "refractive_index": 1.4,
        "priority": 3
    }
    blood_vessel_2 = {
        "name": "blood_vessel_2",
        "structure_type": sp.StructureType.CYLINDER,
        "start_mm": [15, 5, 5],
        "end_mm": [15, 15, 5],
        "radius_mm": 1,
        "mus": 1.2,
        "mua": 0.2,
        "g": 0.9,
        "refractive_index": 1.4,
        "priority": 4
    }
    return [background, muscle_layer, epidermis_layer, blood_vessel_1, blood_vessel_2]

# Set up simulation settings
settings = sp.Settings()
settings[sp.VOLUME_NAME] = "TissuePhantom"
settings[sp.SIMULATION_PATH] = SIMULATION_PATH
settings[sp.VOLUME_EXTENT_MM] = np.array([VOLUME_WIDTH_MM, VOLUME_DEPTH_MM, VOLUME_HEIGHT_MM])
settings[sp.SPACING_MM] = SPACING_MM
settings[sp.WAVELENGTHS] = WAVELENGTHS
settings[sp.ILLUMINATION_TYPE] = sp.IlluminationType.POINT
settings[sp.PULSE_ENERGY_IN_MILLIJOULE] = 50
settings[sp.DETECTOR_POSITIONS_MM] = [[10, 10, 0]]
settings[sp.NUMBER_OF_PHOTONS] = 1e6
settings[sp.STRUCTURES] = create_tissue_structure()

# Create simulation pipeline
pipeline = [
    sp.ModelBasedVolumeCreationAdapter(settings),
    sp.MonteCarloLightTransportAdapter(settings),
    sp.AcousticForwardModelAdapter(settings),
    sp.ReconstructionAdapter(settings),
    sp.LinearUnmixingAdapter(settings)
]

# Run simulation
sp.simulate(pipeline, settings)

# Load results
results = sp.load_data_field(SIMULATION_PATH, sp.RECONSTRUCTED_DATA_FIELD)
unmixing_results = sp.load_data_field(SIMULATION_PATH, sp.LINEAR_UNMIXING_RESULT)

# Visualization
def plot_volume(volume, title, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    plt.figure(figsize=(8, 6))
    plt.imshow(volume[:, :, slice_idx], cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Plot reconstructed data for each wavelength
for idx, wl in enumerate(WAVELENGTHS):
    plot_volume(results[..., idx], f"Reconstructed PA Signal at {wl} nm")

# Plot linear unmixing results
for i, chromophore in enumerate(unmixing_results.dtype.names):
    plot_volume(unmixing_results[chromophore], f"Linear Unmixing: {chromophore}")
```