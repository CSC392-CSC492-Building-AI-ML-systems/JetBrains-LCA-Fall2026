```python
import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.minimum_norm import make_inverse_operator, apply_inverse
import numpy as np

# Load SPM Faces dataset
data_path = spm_face.data_path()
subject = 'subject_01'
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'
event_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D-eve.fif'
subjects_dir = data_path + '/subjects'

# Load raw data
raw = mne.io.read_raw_ctf(raw_fname, preload=True)
raw.filter(1., 40., fir_design='firwin')

# Find events and set up epochs
events = mne.find_events(raw)
event_id = {'faces': 1, 'scrambled': 2}
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

# Fit ICA for artifact removal
ica = ICA(n_components=20, random_state=97, max_iter='auto')
ica.fit(epochs)

# Find and remove EOG/ECG artifacts
eog_inds, scores = ica.find_bads_eog(epochs)
ica.exclude = eog_inds
ecg_inds, scores = ica.find_bads_ecg(epochs)
ica.exclude += ecg_inds
epochs_clean = ica.apply(epochs.copy())

# Average epochs for each condition
evoked_faces = epochs_clean['faces'].average()
evoked_scrambled = epochs_clean['scrambled'].average()

# Compute contrast: faces - scrambled
evoked_contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])

# Estimate noise covariance from baseline
noise_cov = mne.compute_covariance(epochs_clean, tmax=0., method='auto')

# Visualize fields on MEG helmet
evoked_contrast.plot_topomap(times=np.linspace(0.1, 0.2, 5), ch_type='mag')

# Compute forward solution
trans = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D-trans.fif'
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(bem)
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, meg=True, eeg=False)

# Compute inverse operator
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)

# Source reconstruction using dSPM
stc_contrast = apply_inverse(evoked_contrast, inverse_operator, lambda2=1. / 9., method='dSPM')

# Plot the contrast in 3D
brain = stc_contrast.plot(subject=subject, subjects_dir=subjects_dir, initial_time=0.170, hemi='split', views='lat')
```