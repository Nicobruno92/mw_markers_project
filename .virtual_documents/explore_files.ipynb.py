get_ipython().run_line_magic("matplotlib", " qt")
import mne
import matplotlib.pyplot as plt
import utils
import cleaner
from autoreject import AutoReject, get_rejection_threshold
from mne.preprocessing import create_eog_epochs, create_ecg_epochs
from autoreject import AutoReject


# mne.viz.set_3d_backend("notebook")



path = '/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/'
file = 'VP09_crop_rMean_minmarker.set'

raw = mne.io.read_raw_eeglab(path+file, preload = True)



print(raw.info)

raw.set_channel_types({'EXG3': 'eog', 'EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog', 'EXG1': 'ecg', 'EXG7': 'ecg'})

# raw.info['bads'] = ['Pz', 'EXG1', 'EXG2', 'EXG7', 'EXG8', 'Fpz']


#downdsample
# raw_downsampled = raw.copy().resample(sfreq = 250)


raw.plot_psd(fmax = 250, xscale = 'linear')
plt.show()
raw.plot_psd(fmax = 100, xscale = 'log')
plt.show()


lfreq = 1 
hfreq = 40
raw.filter(lfreq, hfreq,
            filter_length='auto', phase='zero', fir_window='hamming',
            fir_design='firwin')


events, event_dict = utils.make_correct_labels(raw)
event_dict


events_correct_go = mne.pick_events(events, include = ['PC/on-task/go/correct/5'])


mne.viz.plot_events(events, event_id = event_dict, sfreq=raw.info['sfreq'],
                          first_samp=raw.first_samp)
plt.show()


#####EPOCHING#####
tmin, tmax = -0.8, 0.8
# rejection_criteria = dict(eeg = 100e-06)
epochs = mne.Epochs(raw, events = events, event_id= event_dict, baseline = (None,None),
                    tmin = tmin, tmax = tmax, picks = ('eeg', 'eog'), preload = True,
#                     reject = rejection_criteria, 
                    verbose = False)


epochs_go_correct = epochs['go/correct']
epochs_go_correct.plot()


print(epochs['SC'])
print(epochs['PC'])


print('Go/Correct: {}'.format(len(epochs['go/correct'])))
print('No-Go/Correct: {}'.format(len(epochs['nogo/correct'])))
print('Go/Incorrect: {}'.format(len(epochs['go/incorrect'])))
print('No-Go/Incorrect: {}'.format(len(epochs['nogo/incorrect'])))


ar = AutoReject()
epochs_clean = ar.fit_transform(epochs)  
reject = get_rejection_threshold(epochs)  


epochs_clean.save(fname = file + '-epo.fif' )


epochs.plot()
plt.show()


evoked_clean = epochs_clean.average()
evoked = epochs.average()

# set_matplotlib_defaults(plt)

fig, axes = plt.subplots(2, 1, figsize=(6, 6))

for ax in axes:
    ax.tick_params(axis='x', which='both', bottom='off', top='off')
    ax.tick_params(axis='y', which='both', left='off', right='off')

ylim = dict(grad=(-170, 200))
# evoked.pick_types(meg='grad', exclude=[])
evoked.plot(exclude=[], axes=axes[0], ylim=ylim, show=False)
axes[0].set_title('Before autoreject')
# evoked_clean.pick_types(meg='grad', exclude=[])
evoked_clean.plot(exclude=[], axes=axes[1], ylim=ylim)
axes[1].set_title('After autoreject')
plt.tight_layout()


ar.get_reject_log(epochs).plot()


n_components = 0.99  # Should normally be higher, like 0.999get_ipython().getoutput("!")
method = 'fastica'
max_iter = 500  # Should normally be higher, like 500 or even 1000get_ipython().getoutput("!")
fit_params = dict(fastica_it=5)
random_state = 42

ica = mne.preprocessing.ICA(n_components=n_components,
                            method=method,
                            max_iter=max_iter,
#                             fit_params=fit_params,
                            random_state=random_state)

ica.fit(raw, decim=1)


# ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
# ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, threshold='auto')

ica.plot_components()


# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw,ch_name = 'Fz', method='correlation',
                                            threshold='auto')
ica.exclude = ecg_indices

# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)

# plot diagnostics
ica.plot_properties(raw, picks=ecg_indices)

# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(raw, show_scrollbars=False)

# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
ica.plot_sources(ecg_evoked)


# ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, reject=None,
#                                                  baseline=(None, -0.2),
#                                                  tmin=-0.5, tmax=0.5)
# ecg_evoked = ecg_epochs.average()
# ecg_inds, ecg_scores = ica.find_bads_ecg(
#     ecg_epochs, method='ctps')


eog_epochs = mne.preprocessing.create_eog_epochs(raw, reject=None,
                                                 baseline=(None, -0.2),
                                                 tmin=-0.5, tmax=0.5)
eog_evoked = eog_epochs.average()
eog_inds, eog_scores = ica.find_bads_eog(
    eog_epochs)

# components_to_exclude = ecg_inds + eog_inds
components_to_exclude = eog_inds
ica.exclude = components_to_exclude


eog_evoked = create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()


ica.plot_scores(eog_scores)
ica.plot_sources(eog_evoked)

ica.plot_properties(epochs_clean)



ica.plot_overlay(eog_evoked)


evoked_sc = epochs_clean['SC/sMW', 'SC/dMW'].average()
evoked_pc = epochs_clean['PC/sMW', 'PC/dMW'].average()
mne.viz.plot_compare_evokeds([evoked_sc, evoked_pc], picks='eeg')


evoked_ot = epochs_clean['PC/on-task'].average()
evoked_mw = epochs_clean['PC/sMW', 'PC/dMW'].average()
mne.viz.plot_compare_evokeds([evoked_ot, evoked_mw], picks='eeg')


evoked_sc_sMW = epochs_clean['SC/sMW'].average()
evoked_sc_dMW = epochs_clean['SC/dMW'].average()
mne.viz.plot_compare_evokeds([evoked_sc_sMW , evoked_sc_dMW], picks='eeg')


get_ipython().run_line_magic("run", " /home/nicolas.bruno/eeg_cleaner/scripts/3_clean_ica.py --path='VP11_crop_rMean_trim10_modmarker6.set-epo.fif' --icaname='auto'")
