import numpy as np
from scipy.io import savemat

import mne
from mne import io
from mne.datasets import sample
from mne.inverse_sparse.mxne_inverse import is_fixed_orient
from mne.inverse_sparse.mxne_inverse import _to_fixed_ori
from sklearn.metrics.pairwise import pairwise_distances
from os.path import join as pjoin

data = 'sample'

data_path = sample.data_path()
cov_fname = pjoin(data_path, 'MEG/sample/sample_audvis-shrunk-cov.fif')
subjects_dir = pjoin(data_path, 'subjects')
condition = 'Left Auditory'

raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
event_id, tmin, tmax = 1, -0.2, 0.5

# Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

# Set up pick list: EEG + MEG - bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443', 'EEG 053']  # bads + 2 more

picks = mne.pick_types(raw.info, meg=True, eeg=True, stim=True, eog=True,
                       exclude='bads')
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=reject)
evoked = epochs.average()
evoked = evoked.pick_types(eeg=True, meg=True)

spacing = 'oct5'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
conductivity = (0.3, 0.006, 0.3)  # for three layers
subject = 'sample'
src = mne.setup_source_space(subject, spacing=spacing,
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)

model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                fname=None, meg=True, eeg=True,
                                mindist=5.0, n_jobs=2)

# convert to surface orientation
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

assert fwd["surf_ori"]
# convert to fixed orientation
assert not is_fixed_orient(fwd)
_to_fixed_ori(fwd)

evoked = epochs.average()
all_ch_names = evoked.ch_names

noise_cov = mne.read_cov(cov_fname)

fwd = mne.pick_channels_forward(fwd, all_ch_names)
fwd_matrix = fwd["sol"]["data"]
n_sensors, n_sources = fwd_matrix.shape

noise_cov = mne.pick_channels_cov(noise_cov, all_ch_names)
noise_cov = mne.cov.regularize(noise_cov, evoked.info, mag=0.01, grad=0.)
cov_reduced = noise_cov['data']

assert noise_cov['names'] == all_ch_names
assert len(noise_cov['data']) == n_sensors


picks = {}
picks["eeg"] = mne.pick_types(evoked.info, meg=False, eeg=True)
picks["mag"] = mne.pick_types(evoked.info, meg='mag', eeg=False)
picks["grad"] = mne.pick_types(evoked.info, meg='grad', eeg=False)

# diagonal covariances
noise_lvls = {}
for s_type in ["grad", "mag", "eeg"]:
    noise_cov_channel_type = noise_cov.data[picks[s_type], :][:, picks[s_type]]
    noise_lvls[s_type] = np.sqrt(np.mean(np.diagonal(noise_cov_channel_type)))

perm = np.concatenate([picks["grad"], picks["mag"], picks["eeg"]])
block_indices = np.cumsum([0, len(picks["grad"]), len(picks["mag"]),
                           len(picks["eeg"])]).astype(np.int32)
distances = pairwise_distances(fwd["source_rr"])

# reorder fwd_matrix so that the channel types are consecutive
fwd_shcl = np.asfortranarray(fwd_matrix[perm])
# as fortran order for faster column access for CD
G_shcl = fwd_shcl.copy(order='F')

savemat('meeg.mat', dict(X=G_shcl,
        sigmas=np.array([noise_lvls["grad"], noise_lvls["mag"], noise_lvls["eeg"]]),
        block_indices=block_indices,
        distances=distances))
