```python
import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
import numpy as np

# 1. Load SPM Faces dataset
data_path = spm_face.data_path()
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'
event_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'
subjects_dir = data_path + '/subjects'
subject = 'spm'

# 2. Load and filter data
raw = mne.io.read_raw_ctf(raw_fname, preload=True)
raw.filter(1., 40., fir_design='firwin')

# 3. Set up events and epochs
events = mne.find_events(raw, stim_channel='UPPT001')
event_id = {'faces': 1, 'scrambled': 2}
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

# 4. Fit ICA
ica = ICA(n_components=20, random_state=97, max_iter='auto')
ica.fit(epochs)

# 5. Find and remove major artifacts (EOG/ECG)
eog_epochs = create_eog_epochs(raw)
eog_inds, scores = ica.find_bads_eog(eog_epochs)
ica.exclude = eog_inds

ecg_epochs = create_ecg_epochs(raw)
ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)
ica.exclude += ecg_inds

epochs_clean = ica.apply(epochs.copy())

# 6. Average epochs for each condition
evoked_faces = epochs_clean['faces'].average()
evoked_scrambled = epochs_clean['scrambled'].average()

# 7. Compute and visualize the contrast
evoked_contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])
evoked_contrast.plot_joint(title='Faces - Scrambled Contrast')

# 8. Estimate noise covariance
noise_cov = mne.compute_covariance(epochs_clean, tmax=0., method='auto')

# 9. Visualize fields on MEG helmet
evoked_contrast.plot_topomap(times=np.linspace(0.1, 0.2, 5), title='Contrast Topomaps')

# 10. Compute forward solution
trans = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D-trans.fif'
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model(subject=subject, ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(bem)
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, meg=True, eeg=False)

# 11. Compute inverse solution (dSPM)
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
stc_contrast = apply_inverse(evoked_contrast, inverse_operator, lambda2=1. / 9., method='dSPM')

# 12. Plot the contrast in 3D
brain = stc_contrast.plot(subject=subject, subjects_dir=subjects_dir, hemi='split',
                         initial_time=0.170, time_unit='s', title='Faces - Scrambled (dSPM)')
```