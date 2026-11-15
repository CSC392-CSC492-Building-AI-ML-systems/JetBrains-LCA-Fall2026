```python
import mne
from mne.datasets import spm_face
from mne.preprocessing import ICA, create_eog_epochs
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_alignment

# 1. Load SPM Faces dataset
data_path = spm_face.data_path()
raw_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.ds'
event_fname = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D.eve'

raw = mne.io.read_raw_ctf(raw_fname, preload=True)
events = mne.read_events(event_fname)

# 2. Filter data
raw.filter(1., 40., fir_design='firwin')

# 3. Set up epochs
event_id = {'faces': 1, 'scrambled': 2}
tmin, tmax = -0.2, 0.5
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=(None, 0), preload=True)

# 4. Fit ICA
ica = ICA(n_components=20, random_state=97, max_iter='auto')
ica.fit(epochs)

# 5. Find and remove EOG artifacts
eog_epochs = create_eog_epochs(raw, ch_name='EEG057')  # SPM Faces has EEG057 as EOG
eog_inds, scores = ica.find_bads_eog(eog_epochs)
ica.exclude = eog_inds
epochs_clean = ica.apply(epochs.copy())

# 6. Average epochs
evoked_faces = epochs_clean['faces'].average()
evoked_scrambled = epochs_clean['scrambled'].average()

# 7. Compute contrast
evoked_contrast = mne.combine_evoked([evoked_faces, evoked_scrambled], weights=[1, -1])

# 8. Estimate noise covariance
noise_cov = mne.compute_covariance(epochs_clean, tmax=0., method='auto')

# 9. Visualize fields on MEG helmet
evoked_contrast.plot_topomap(times=0.17, ch_type='mag', title='Contrast: faces - scrambled')

# 10. Compute forward model
subjects_dir = data_path + '/subjects'
trans = data_path + '/MEG/spm/SPM_CTF_MEG_example_faces1_3D-trans.fif'
src = mne.setup_source_space('spm', spacing='oct6', subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model(subject='spm', ico=4, conductivity=(0.3,), subjects_dir=subjects_dir)
bem_sol = mne.make_bem_solution(bem)
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem_sol, meg=True, eeg=False)

# 11. Compute inverse solution (dSPM)
inverse_operator = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
stc = apply_inverse(evoked_contrast, inverse_operator, lambda2=1. / 9., method='dSPM')

# 12. Plot the contrast in 3D
brain = stc.plot(subject='spm', subjects_dir=subjects_dir, initial_time=0.17, hemi='both',
                 time_viewer=False, title='dSPM: faces - scrambled')
```