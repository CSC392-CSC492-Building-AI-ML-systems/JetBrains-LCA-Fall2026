```python
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
from simpa.utils import Tags
from simpa.linear_unmixing import LinearUnmixing

# Set global simulation parameters
VOLUME_WIDTH_MM = 30
VOLUME_HEIGHT_MM = 30
VOLUME_DEPTH_MM = 10
SPACING_MM = 0.5
WAVELENGTHS = [700, 800, 900]  # nm

# Simulation path
SIMULATION_PATH = "simpa_tissue_simulation"

# Define simulation settings
settings = {
    Tags.SIMULATION_PATH: SIMULATION_PATH,
    Tags.VOLUME_NAME: "tissue_volume",
    Tags.RANDOM_SEED: 42,
    Tags.WAVELENGTHS: WAVELENGTHS,
    Tags.SPACING_MM: SPACING_MM,
    Tags.DIM_VOLUME_X_MM: VOLUME_WIDTH_MM,
    Tags.DIM_VOLUME_Y_MM: VOLUME_HEIGHT_MM,
    Tags.DIM_VOLUME_Z_MM: VOLUME_DEPTH_MM,
    Tags.STRUCTURE_DEFINITION: "custom_tissue_structure",
    Tags.SIMULATION_MODULES: [
        Tags.OPTICAL_MODEL,
        Tags.ACOUSTIC_MODEL,
        Tags.RECONSTRUCTION_MODULE
    ],
    Tags.OPTICAL_MODEL: Tags.MCML_MODEL,
    Tags.ACOUSTIC_MODEL: Tags.K_WAVE_LINEAR,
    Tags.RECONSTRUCTION_MODULE: Tags.FILTERED_BACK_PROJECTION,
    Tags.DEVICE: sp.devices.ImaginaryLinearArrayDevice(angular_coverage_deg=90, number_detector_elements=128)
}

# Function to create the tissue structure
def create_tissue_structure(settings):
    structures = []

    # Background
    structures.append({
        Tags.PRIORITY: 0,
        Tags.STRUCTURE_TYPE: Tags.BACKGROUND_STRUCTURE,
        Tags.MOLECULE_COMPOSITION: {
            Tags.MUSCLE: 1.0
        },
        Tags.STRUCTURE_START_MM: [0, 0, 0],
        Tags.STRUCTURE_END_MM: [VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, VOLUME_DEPTH_MM]
    })

    # Muscle layer
    structures.append({
        Tags.PRIORITY: 1,
        Tags.STRUCTURE_TYPE: Tags.LAYER_STRUCTURE,
        Tags.MOLECULE_COMPOSITION: {
            Tags.MUSCLE: 1.0
        },
        Tags.STRUCTURE_START_MM: [0, 0, 2],
        Tags.STRUCTURE_END_MM: [VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, 8]
    })

    # Epidermis layer
    structures.append({
        Tags.PRIORITY: 2,
        Tags.STRUCTURE_TYPE: Tags.LAYER_STRUCTURE,
        Tags.MOLECULE_COMPOSITION: {
            Tags.EPIDERMIS: 1.0
        },
        Tags.STRUCTURE_START_MM: [0, 0, 0],
        Tags.STRUCTURE_END_MM: [VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, 0.5]
    })

    # Blood vessel 1
    structures.append({
        Tags.PRIORITY: 3,
        Tags.STRUCTURE_TYPE: Tags.CYLINDER_STRUCTURE,
        Tags.MOLECULE_COMPOSITION: {
            Tags.BLOOD: 1.0
        },
        Tags.CYLINDER_RADIUS_MM: 1.0,
        Tags.CYLINDER_AXIS_START_MM: [10, 15, 3],
        Tags.CYLINDER_AXIS_END_MM: [10, 15, 7]
    })

    # Blood vessel 2
    structures.append({
        Tags.PRIORITY: 4,
        Tags.STRUCTURE_TYPE: Tags.CYLINDER_STRUCTURE,
        Tags.MOLECULE_COMPOSITION: {
            Tags.BLOOD: 1.0
        },
        Tags.CYLINDER_RADIUS_MM: 1.5,
        Tags.CYLINDER_AXIS_START_MM: [20, 10, 4],
        Tags.CYLINDER_AXIS_END_MM: [20, 10, 8]
    })

    return structures

# Register the tissue structure creation function
sp.TissueLibrary().register_custom_structure("custom_tissue_structure", create_tissue_structure)

# Run the simulation
pipeline = sp.SimulationPipeline(settings)
pipeline.run()

# Load the simulation results
sim_data = sp.load_data_field(SIMULATION_PATH, Tags.RECONSTRUCTED_DATA, wavelength=WAVELENGTHS[0])

# Linear unmixing
unmixer = LinearUnmixing(SIMULATION_PATH, wavelengths=WAVELENGTHS)
unmixing_result = unmixer.unmix()

# Visualization
def plot_volume(volume, title, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    plt.figure(figsize=(6, 6))
    plt.imshow(volume[:, :, slice_idx], cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Plot reconstructed data for each wavelength
for wl in WAVELENGTHS:
    data = sp.load_data_field(SIMULATION_PATH, Tags.RECONSTRUCTED_DATA, wavelength=wl)
    plot_volume(data, f"Reconstructed Data at {wl} nm")

# Plot linear unmixing results
for chromophore, volume in unmixing_result.items():
    plot_volume(volume, f"Unmixed: {chromophore}")
```