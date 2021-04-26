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
    epochs = mne.read_epochs(folder +  participant + '_' + epoch_type + '_' +  'ar_subtracted_epo.fif') 
    epochs.info['description'] = 'biosemi/64' #necessary for wSMI 
    epochs =  epochs.pick_types(eeg = True) #EOGs break everything\
    
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs['PC/on-task/correct/go'], 0, 0.6, target))
        df['electrode'] = np.arange(0,64)

        df_ot = df_ot.append(df)
    except:
        pass
    
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs['PC/dMW/correct/go'], 0, 0.6, target))
        df['electrode'] = np.arange(0,64)

        df_mw = df_mw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs['PC/sMW/correct/go'], 0, 0.6, target))
        df['electrode'] = np.arange(0,64)

        df_mw =  df_mw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs['SC/dMW/correct/go'], 0, 0.6, target))
        df['electrode'] = np.arange(0,64)

        df_dmw = df_dmw.append(df)
    except:
        pass
    try: 
        df = pd.DataFrame.from_dict(all_markers(epochs['SC/sMW/correct/go'], 0, 0.6, target))
        df['electrode'] = np.arange(0,64)

        df_smw =  df_smw.append(df)
    except:
        pass
    


markers = ['wSMI', 'p_e', 'k', 'b', 'b_n', 'g', 'g_n', 't', 't_n',
       'd', 'd_n', 'a','a_n', 'CNV', 'P1', 'P3a', 'P3b']

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
    


markers = ['wSMI', 'p_e', 'k', 'b', 'b_n', 'g', 'g_n', 't', 't_n',
       'd', 'd_n', 'a','a_n', 'CNV', 'P1', 'P3a', 'P3b']

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
    
