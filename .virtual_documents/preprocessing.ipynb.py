get_ipython().run_line_magic("matplotlib", " qt")
import mne
import utils
import matplotlib.pyplot as plt

# from cleaner.report import create_ica_report 
# from cleaner import reject

# from nice_ext.algorithms.adaptive import _adaptive_egi

# from autoreject import AutoReject, get_rejection_threshold


epoch_type = 'evoked'
# epoch_type = 'pseudo-rs'


all_participants = ['VP07','VP08','VP09', 'VP10','VP11','VP12','VP13','VP14','VP18','VP19','VP20','VP23','VP24','VP25','VP26','VP27','VP28','VP29','VP30','VP31','VP32','VP33','VP35','VP36','VP37']
participant = all_participants[1]

# path = '/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/'
path = '/Users/nicobruno/ownCloud/MW_eeg_data/minmarker/'

folder = path + participant +'/'
file = participant + '_crop_rMean_minmarker.set'

raw = mne.io.read_raw_eeglab(folder+file, preload = True, verbose = False)


raw.set_channel_types({'EXG3': 'eog', 'EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog'})

raw.info['bads'] = ['EXG1', 'EXG2', 'EXG7', 'EXG8']

print(raw.info)


raw_downsampled = raw.copy().resample(sfreq=250)


hpass = 0.5
lpass = 45

raw_filtered = raw_downsampled.copy().filter(l_freq=hpass, h_freq=lpass)


fig, ax = plt.subplots(2)

raw_downsampled.plot_psd(ax=ax[0], show=False)
raw_filtered.plot_psd(ax=ax[1], show=False)

ax[0].set_title('PSD before filtering')
ax[1].set_title('PSD after filtering')
ax[1].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)
plt.show()


raw_filtered.plot()


### Save filtered data
raw_filtered.save(folder + participant + 'filt_raw.fif', overwrite = True)


raw_filtered = mne.io.read_raw(folder + participant + 'filt_raw.fif')


events, event_dict = utils.make_correct_labels(raw_filtered)


epochs = utils.create_epochs(epoch_type = epoch_type, raw = raw_filtered, events =  events, event_id = event_dict, save = True)


reject_criteria = dict(eeg=150e-6)       # 150 µV
epochs.drop_bad(reject=reject_criteria)
epochs.plot_drop_log()


epochs.plot()


epochs.plot_drop_log()


epochs.save(folder + participant + epoch_type + 'clean_epo.fif', overwrite = True)


epochs_clean = mne.read_epochs(folder + participant + epoch_type + 'clean_epo.fif')

# epochs_clean.info['bads'].append(['P2', 'PO4'])
epochs_clean.info


n_components = 0.99  # Should normally be higher, like 0.999get_ipython().getoutput("!")
method = 'fastica'
max_iter = 512  # Should normally be higher, like 500 or even 1000get_ipython().getoutput("!")
fit_params = dict(fastica_it=5)
random_state = 42

ica = mne.preprocessing.ICA(n_components=n_components,
                            method=method,
                            max_iter=max_iter,
#                             fit_params=fit_params,
                            random_state=random_state)

ica.fit(epochs_clean)


# layout = mne.channels.make_eeg_layout(epochs.info)
ica.plot_components(inst = epochs_clean,picks=range(15))


# reject(folder + participant + '-ica-epo.fif', ica)
ica.plot_sources(epochs_clean, block=True)
# ica.plot_properties(epochs)


ica.plot_overlay(raw_filtered, exclude=[1])






