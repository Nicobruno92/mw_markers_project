get_ipython().run_line_magic("matplotlib", " qt")
import mne
import utils
import matplotlib.pyplot as plt



epoch_type = 'evoked'
# epoch_type = 'pseudo-rs'


all_participants = ['VP07','VP08','VP09', 'VP10','VP11','VP12','VP13','VP14','VP18','VP19','VP20','VP23','VP24','VP25','VP26','VP27','VP28','VP29','VP30','VP31','VP32','VP33','VP35','VP36','VP37']
participant = all_participants[2]

# path = '/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/' #icm-linux
path = '/Users/nicobruno/ownCloud/MW_eeg_data/minmarker/' #nico-mac

folder = path + participant +'/'
file = participant + '_crop_rMean_minmarker.set'



epochs_ar = mne.read_epochs(folder + participant + epoch_type + 'ar-rereferenced_epo.fif', verbose = False)['go/correct']
epochs_manual = mne.read_epochs(folder + participant + epoch_type + 'rereferenced_epo.fif', verbose = False)['go/correct']


P3_electrodes =['FCz', 'FC1','FC2', 'C1', 'Cz', 'C2', 'CP1','CPz','CP2','P1', 'Pz','P2', 'PO3','POz','PO4' ] 
epochs_ar = epochs_ar.pick_channels(P3_electrodes)
epochs_manual = epochs_manual.pick_channels(P3_electrodes)


evoked_ar = epochs_ar.average()
evoked_manual = epochs_manual.average()


print(len(epochs_ar), len(epochs_manual))


for combine in ('mean', 'median', 'gfp'):
    mne.viz.plot_compare_evokeds([evoked_ar, evoked_manual], picks='eeg', combine=combine)


evoked_ar.plot()
evoked_manual.plot()



