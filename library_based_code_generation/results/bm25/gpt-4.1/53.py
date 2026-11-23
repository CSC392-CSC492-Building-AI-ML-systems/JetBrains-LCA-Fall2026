```python
import simpa as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# Set global simulation parameters
VOLUME_WIDTH_MM = 20
VOLUME_HEIGHT_MM = 10
VOLUME_DEPTH_MM = 20
SPACING_MM = 0.5
WAVELENGTHS = [700, 800, 900]  # nm

# Define simulation path
SIMULATION_PATH = "simpa_tissue_simulation"
os.makedirs(SIMULATION_PATH, exist_ok=True)

# Define global settings
settings = {
    "wavelengths": WAVELENGTHS,
    "volume_name": "TissueVolume",
    "spacing_mm": SPACING_MM,
    "volume_extent_mm": np.array([VOLUME_WIDTH_MM, VOLUME_HEIGHT_MM, VOLUME_DEPTH_MM]),
    "output_path": SIMULATION_PATH,
    "device": sp.devices.ImalacDevice(),
    "number_of_threads": 4,
    "simpa_output_path": SIMULATION_PATH,
    "random_seed": 42,
    "noise_model": None,
    "do_file_compression": False,
    "save_type": "hdf5"
}

# Function to create the tissue structure
def create_tissue_structure():
    background = {
        "name": "background",
        "structure_type": sp.StructureType.BACKGROUND,
        "optical_properties": sp.TissueLibrary.air()
    }
    muscle_layer = {
        "name": "muscle",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 2,
        "z_end_mm": 8,
        "optical_properties": sp.TissueLibrary.muscle()
    }
    epidermis_layer = {
        "name": "epidermis",
        "structure_type": sp.StructureType.LAYER,
        "z_start_mm": 0,
        "z_end_mm": 0.5,
        "optical_properties": sp.TissueLibrary.epidermis()
    }
    blood_vessel_1 = {
        "name": "blood_vessel_1",
        "structure_type": sp.StructureType.CYLINDER,
        "center_mm": [10, 5, 5],
        "radius_mm": 1,
        "height_mm": 6,
        "direction": [0, 1, 0],
        "optical_properties": sp.TissueLibrary.blood(oxygenation=0.9)
    }
    blood_vessel_2 = {
        "name": "blood_vessel_2",
        "structure_type": sp.StructureType.CYLINDER,
        "center_mm": [15, 5, 10],
        "radius_mm": 0.8,
        "height_mm": 5,
        "direction": [0, 1, 0],
        "optical_properties": sp.TissueLibrary.blood(oxygenation=0.7)
    }
    return [background, muscle_layer, epidermis_layer, blood_vessel_1, blood_vessel_2]

# Add tissue structure to settings
settings["tissue_structures"] = create_tissue_structure()

# Run the simulation
sp.run_simulation(settings)

# Linear unmixing
unmixing_settings = settings.copy()
unmixing_settings["unmixing_method"] = "linear"
unmixing_settings["unmixing_wavelengths"] = WAVELENGTHS
unmixing_settings["unmixing_molecules"] = ["HbO2", "Hb"]

unmixing_module = sp.modules.unmixing.LinearUnmixing(unmixing_settings)
unmixing_module.run()

# Load and visualize results
sim_data = sp.load_data_field(SIMULATION_PATH, "TissueVolume", sp.Tags.DATA_FIELD)
unmixed_data = sp.load_data_field(SIMULATION_PATH, "TissueVolume", sp.Tags.UNMIXING_RESULT)

# Visualize the central slice for each wavelength
fig, axs = plt.subplots(1, len(WAVELENGTHS), figsize=(15, 5))
for i, wl in enumerate(WAVELENGTHS):
    axs[i].imshow(sim_data[wl][:, :, sim_data[wl].shape[2] // 2], cmap='hot')
    axs[i].set_title(f"Simulated PA at {wl} nm")
    axs[i].axis('off')
plt.tight_layout()
plt.show()

# Visualize linear unmixing results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(unmixed_data["HbO2"][:, :, unmixed_data["HbO2"].shape[2] // 2], cmap='Reds')
axs[0].set_title("HbO2 Unmixing")
axs[0].axis('off')
axs[1].imshow(unmixed_data["Hb"][:, :, unmixed_data["Hb"].shape[2] // 2], cmap='Blues')
axs[1].set_title("Hb Unmixing")
axs[1].axis('off')
plt.tight_layout()
plt.show()
```