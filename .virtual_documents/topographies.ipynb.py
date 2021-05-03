# get_ipython().run_line_magic("matplotlib", " qt")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne 

from utils import all_markers


epoch_type = 'evoked'
# epoch_type = 'pseudo-rs'


all_participants = ['VP07','VP08','VP09', 'VP10','VP11','VP12','VP13','VP14','VP18','VP19','VP20','VP22','VP23','VP24','VP25','VP26','VP27','VP28','VP29','VP30','VP31','VP32','VP33','VP35','VP36','VP37']


# path = '/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/' #icm-linux
path = '/Users/nicobruno/ownCloud/MW_eeg_data/minmarker/' #nico-mac


target = 'topography'

df_mw = pd.DataFrame()
df_ot = pd.DataFrame()

df_smw = pd.DataFrame()
df_dmw = pd.DataFrame()

for i in all_participants:
    participant = i
    
    print('')
    print('#########################################')
    print('Computing markers for participant {}'.format(participant))
    print('#########################################')
    print('')
    

    folder = path + participant +'/'

    
    df_markers = pd.DataFrame()
    df_markers['participant'] = participant
    
    #############################
    #### With ERP SUBTRACTED ####
    #############################
    epochs_subtracted = mne.read_epochs(folder +  participant + '_' + epoch_type + '_' +  'ar_subtracted_epo.fif') 
    epochs_subtracted.info['description'] = 'biosemi/64' #necessary for wSMI 
    epochs_subtracted =  epochs_subtracted.pick_types(eeg = True) #EOGs break everything\
    

    #############################
    ####       With ERP      ####
    #############################
    epochs_erp = mne.read_epochs(folder +  participant + '_' + epoch_type + '_' +  'ar_rereferenced_epo.fif') 
    epochs_erp.info['description'] = 'biosemi/64' #necessary for wSMI 
    epochs_erp =  epochs_erp.pick_types(eeg = True) #EOGs break everything\
    
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs_subtracted['PC/on-task/correct/go'], 0, 0.6, target, epochs_erp =epochs_erp['PC/on-task/correct/go']))
        df['electrode'] = np.arange(0,64)
        df['participant'] = i

        df_ot = df_ot.append(df)
    except:
        pass
    
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs_subtracted['PC/dMW/correct/go'], 0, 0.6, target, epochs_erp =epochs_erp['PC/dMW/correct/go']))
        df['electrode'] = np.arange(0,64)
        df['participant'] = i

        df_mw = df_mw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs_subtracted['PC/sMW/correct/go'], 0, 0.6, target, epochs_erp =epochs_erp['PC/sMW/correct/go']))
        df['electrode'] = np.arange(0,64)
        df['participant'] = i

        df_mw =  df_mw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs_subtracted['SC/dMW/correct/go'], 0, 0.6, target, epochs_erp =epochs_erp['SC/dMW/correct/go']))
        df['electrode'] = np.arange(0,64)
        df['participant'] = i

        df_dmw = df_dmw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs_subtracted['SC/sMW/correct/go'], 0, 0.6, target, epochs_erp =epochs_erp['SC/sMW/correct/go']))
        df['electrode'] = np.arange(0,64)
        df['participant'] = i

        df_smw =  df_smw.append(df)
    except:
        pass
    


markers = ['wSMI_1', 'wSMI_2', 'wSMI_4', 'wSMI_8', 'p_e_1', 'p_e_2',
       'p_e_4', 'p_e_8', 'k', 'b', 'b_n', 'g', 'g_n', 't', 't_n',
       'd', 'd_n', 'a_n', 'a', 'CNV', 'P1', 'P3a', 'P3b']

epochs = epochs_subtracted

df_ot_agg = df_ot.groupby('electrode').mean()
df_mw_agg = df_mw.groupby('electrode').mean()

df_topo = df_mw_agg
df_topo = df_topo.append(df_ot_agg)

df_topo = df_topo.groupby('electrode').diff().dropna()

for marker in markers:
    
    fig, ax = plt.subplots(1,3)

    fig.suptitle(marker)

    im, _ = mne.viz.plot_topomap(df_ot_agg[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_ot_agg[marker].values), vmax = np.nanmax(df_ot_agg[marker].values), axes = ax[0], show = False)
    ax[0].set_title('OT')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_ot_agg[marker].values), np.nanmax(df_ot_agg[marker].values)))
    cbar.ax.tick_params(labelsize=8)

    im, _ = mne.viz.plot_topomap(df_mw_agg[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_mw_agg[marker].values), vmax = np.nanmax(df_mw_agg[marker].values), axes = ax[1], show = False)
    ax[1].set_title('MW')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_mw_agg[marker].values), np.nanmax(df_mw_agg[marker].values)))
    cbar.ax.tick_params(labelsize=8)

    im, _ = mne.viz.plot_topomap(df_topo[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_topo[marker].values), vmax = np.nanmax(df_topo[marker].values), axes = ax[2], show= False)
    ax[2].set_title('OT-MW')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_topo[marker].values), np.nanmax(df_topo[marker].values)))
    cbar.ax.tick_params(labelsize=8)


    plt.show()
    


df_smw_agg = df_smw.groupby('electrode').mean()
df_dmw_agg = df_dmw.groupby('electrode').mean()

df_topo = df_dmw_agg
df_topo = df_topo.append(df_smw_agg)

df_topo = df_topo.groupby('electrode').diff().dropna()

for marker in markers:
    
    fig, ax = plt.subplots(1,3)

    fig.suptitle(marker)

    im, _ = mne.viz.plot_topomap(df_smw_agg[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_smw_agg[marker].values), vmax = np.nanmax(df_smw_agg[marker].values), axes = ax[0], show = False)
    ax[0].set_title('sMW')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_smw_agg[marker].values), np.nanmax(df_smw_agg[marker].values)))
    cbar.ax.tick_params(labelsize=8)

    im, _ = mne.viz.plot_topomap(df_dmw_agg[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_dmw_agg[marker].values), vmax = np.nanmax(df_dmw_agg[marker].values), axes = ax[1], show = False)
    ax[1].set_title('dMW')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_dmw_agg[marker].values), np.nanmax(df_dmw_agg[marker].values)))
    cbar.ax.tick_params(labelsize=8)

    im, _ = mne.viz.plot_topomap(df_topo[marker].values, pos = epochs.info, 
                         image_interp='nearest', outlines='head', sensors=True,
                         vmin = np.nanmin(df_topo[marker].values), vmax = np.nanmax(df_topo[marker].values), axes = ax[2], show= False)
    ax[2].set_title('sMW-dMW')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5get_ipython().run_line_magic("",", " pad=0.05)")
    cbar = plt.colorbar(im, cax=cax, ticks=(np.nanmin(df_topo[marker].values), np.nanmax(df_topo[marker].values)))
    cbar.ax.tick_params(labelsize=8)


    plt.show()
    


from mne.stats import spatio_temporal_cluster_test
from mne.channels import find_ch_adjacency
from scipy.sparse import spmatrix


adjacency, ch_names = find_ch_adjacency(epochs_subtracted.info, 'eeg')
# adjacency = spmatrix(adjacency)


plt.imshow(adjacency.toarray(), cmap='gray', origin='lower',
           interpolation='nearest')
plt.xlabel('{} Magnetometers'.format(len(ch_names)))
plt.ylabel('{} Magnetometers'.format(len(ch_names)))
plt.title('Between-sensor adjacency')


cluster_p = []
for marker in markers:
    print(marker)

    dmw = df_dmw.pivot(index = 'participant',columns = 'electrode')[marker].values
    dmw = dmw.reshape( (len(dmw),1,64) )
    smw = df_smw.pivot(index = 'participant',columns = 'electrode')[marker].values
    smw = smw.reshape( (len(smw),1,64) )

    contrast = [dmw, smw]
    
    threshold = 2
    T_obs, clusters, cluster_p_values, H0 = \
    spatio_temporal_cluster_test(contrast, n_permutations=1000,
                             threshold=threshold, tail=1, n_jobs=1,
                             adjacency= adjacency,
                             out_type='mask')
    cluster_p.append(cluster_p_values)
    


cluster_p


times = epochs_erp.times
plt.close('all')
plt.subplot(211)
# plt.title('Channel : ' + channel)
plt.plot(times, contrast[0].mean(axis=0) - contrast[1].mean(axis=0),
         label="ERF Contrast (Event 1 - Event 2)")
plt.ylabel("MEG (T / m)")
plt.legend()
plt.subplot(212)
for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = plt.axvspan(times[c.start], times[c.stop - 1],
                        color='r', alpha=0.3)
    else:
        plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                    alpha=0.3)
hf = plt.plot(times, T_obs, 'g')
plt.legend((h, ), ('cluster p-value < 0.05', ))
plt.xlabel("time (ms)")
plt.ylabel("f-values")
plt.show()
