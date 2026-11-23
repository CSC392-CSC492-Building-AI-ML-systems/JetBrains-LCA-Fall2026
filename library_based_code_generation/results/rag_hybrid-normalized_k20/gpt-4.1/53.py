```python
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt

# Global simulation parameters
VOLUME_WIDTH_MM = 20
VOLUME_HEIGHT_MM = 10
VOLUME_DEPTH_MM = 20
SPACING_MM = 0.2
WAVELENGTHS = [700, 800, 900]  # nm

# Simulation settings
settings = {
    "volume_name": "tissue_simulation",
    "volume_dimensions_mm": np.asarray([VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, VOLUME_DEPTH_MM]),
    "spacing_mm": SPACING_MM,
    "wavelengths": WAVELENGTHS,
    "output_path": "simpa_output",
    "device": "cpu",
    "number_of_photons": 1e6,
    "do_file_compression": True,
    "simulate_deformed_mesh": False,
    "noise_factor": 0.0,
    "random_seed": 42,
    "save_type": "hdf5"
}

# Function to create the tissue structure
def create_tissue_structure():
    background = {
        "name": "background",
        "structure_type": sp.StructureType.BACKGROUND,
        "optical_properties": sp.TissueLibrary.muscle(),
    }
    muscle_layer = {
        "name": "muscle_layer",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 0,
        "z_end_mm": 7,
        "optical_properties": sp.TissueLibrary.muscle(),
    }
    epidermis_layer = {
        "name": "epidermis_layer",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 7,
        "z_end_mm": 8,
        "optical_properties": sp.TissueLibrary.epidermis(),
    }
    blood_vessel_1 = {
        "name": "blood_vessel_1",
        "structure_type": sp.StructureType.CYLINDER,
        "center_mm": [10, 5, 4],
        "radius_mm": 1,
        "height_mm": 6,
        "direction": [0, 0, 1],
        "optical_properties": sp.TissueLibrary.blood(oxygenation=0.9),
    }
    blood_vessel_2 = {
        "name": "blood_vessel_2",
        "structure_type": sp.StructureType.CYLINDER,
        "center_mm": [15, 5, 6],
        "radius_mm": 0.8,
        "height_mm": 4,
        "direction": [0, 0, 1],
        "optical_properties": sp.TissueLibrary.blood(oxygenation=0.7),
    }
    return [background, muscle_layer, epidermis_layer, blood_vessel_1, blood_vessel_2]

# Create simulation pipeline
pipeline = [
    sp.TissueVolumeCreatorModule(settings),
    sp.OpticalForwardModule(settings),
    sp.AcousticForwardModule(settings),
    sp.ReconstructionModule(settings)
]

# Create the simulation object
simulation = sp.Simulation(settings=settings,
                           pipeline_elements=pipeline,
                           tissue_structure=create_tissue_structure())

# Run the simulation
simulation.run()

# Linear unmixing
unmixing = sp.TestLinearUnmixing(settings)
unmixing.run()

# Load results
sim_data = sp.load_data_field(settings["output_path"], settings["volume_name"], sp.Tags.SIMULATION_OUTPUT)
unmixed_data = sp.load_data_field(settings["output_path"], settings["volume_name"], sp.Tags.UNMIXING_OUTPUT)

# Visualization
def visualize_volume(volume, title, slice_idx=None):
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2
    plt.figure(figsize=(8, 6))
    plt.imshow(volume[:, :, slice_idx], cmap='hot')
    plt.title(title)
    plt.colorbar()
    plt.show()

# Visualize simulation results for each wavelength
for idx, wl in enumerate(WAVELENGTHS):
    visualize_volume(sim_data[..., idx], f"Simulated PA Signal at {wl} nm")

# Visualize linear unmixing results
for i, chromophore in enumerate(unmixing.chromophore_names):
    visualize_volume(unmixed_data[..., i], f"Unmixed {chromophore}")
```