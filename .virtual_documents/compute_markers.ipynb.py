get_ipython().run_line_magic("matplotlib", " qt")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne 
from nice.markers import (KolmogorovComplexity, TimeLockedContrast, PowerSpectralDensityEstimator, 
                          PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy, TimeLockedTopography, ContingentNegativeVariation)

import pycsd

from utils import make_str_label, all_markers


def all_markers(epochs, tmin, tmax, target, epochs_erp = epochs):
        """
        Computes all ther markers for given epochs.
        epochs: the epochs from which to compute the markers
        tmin: min time for computing markers 
        tmax: max time to compute markers
        target: reduction target, epochs or topography
        epochs_erp: if a different preprocessed epochs are used for computing ERP
        
        Evoked markers have already defined times
        """
        from scipy.stats import trim_mean
        
        def trim_mean80(a, axis=0):
            return trim_mean(a, proportiontocut=.1, axis=axis)       

        # =============================================================================
        # SPECTRAL MARKERS
        # =============================================================================
          #PowerSpectralDensityL
        psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)
        base_psd = PowerSpectralDensityEstimator(
            psd_method='welch', tmin=tmin, tmax=tmax, fmin=1., fmax=45.,
            psd_params=psds_params, comment='default')
        


        ###alpha normalized###
        alpha = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=True, comment='alphan')
        alpha.fit(epochs)

        reduction_func = [{'axis': 'frequency', 'function': np.sum},
             {'axis': 'channels', 'function': np.mean},
             {'axis': 'epochs', 'function': trim_mean80}]
        ###alpha normalized###
        alpha = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=True, comment='alpha')
        alpha.fit(epochs)
        dataalpha_n = alpha._reduce_to(reduction_func, target=target, picks=None)

        #alpha
        alpha = PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=13.,normalize=False, comment='alpha')
        alpha.fit(epochs)
        dataalpha = alpha._reduce_to(reduction_func, target=target, picks=None)

        #delta normalized
        delta = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,normalize=True, comment='delta')
        delta.fit(epochs)
        datadelta_n = delta._reduce_to(reduction_func, target=target, picks=None)


        #delta
        delta = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4,normalize=False, comment='delta')
        delta.fit(epochs)
        datadelta = delta._reduce_to(reduction_func, target=target, picks=None)

        #theta normalized
        theta = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,normalize=True, comment='theta')
        theta.fit(epochs)
        datatheta_n = theta._reduce_to(reduction_func, target=target, picks=None)


        #theta
        theta = PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8,normalize=False, comment='theta')
        theta.fit(epochs)
        datatheta = theta._reduce_to(reduction_func, target=target, picks=None)

        #gamma normalized
        gamma = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,normalize=True, comment='gamma')
        gamma.fit(epochs)
        datagamma_n = gamma._reduce_to(reduction_func, target=target, picks=None)


        #gamma
        gamma = PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45,normalize=False, comment='theta')
        gamma.fit(epochs)
        datagamma = gamma._reduce_to(reduction_func, target=target, picks=None)

        #beta normalized
        beta = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30.,normalize=True, comment='beta')
        beta.fit(epochs)
        databetaa_n = beta._reduce_to(reduction_func, target=target, picks=None)


        #beta
        beta = PowerSpectralDensity(estimator=base_psd, fmin=13., fmax=30,normalize=False, comment='beta')
        beta.fit(epochs)
        databeta = beta._reduce_to(reduction_func, target=target, picks=None)


        # =============================================================================
        # INFORMATION THEORY MARKERS
        # =============================================================================

        komplexity = KolmogorovComplexity(tmin=tmin, tmax=tmax, backend='openmp')
        komplexity.fit(epochs)
        komplexityobject=komplexity.data_ ###Object to save, number of channels*number of epochs, it's ndarray
        reduction_func= [{'axis': 'channels', 'function': np.mean},
             {'axis': 'epochs', 'function': trim_mean80}]


        datakomplexity = komplexity._reduce_to(reduction_func, target=target, picks=None)

        
        
        
        ### Permuttion entropy ###
        p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=1)
        p_e.fit(epochs)
        p_eobject = p_e.data_
        datap_e1 = p_e._reduce_to(reduction_func, target=target, picks=None)
        
        ### Permuttion entropy ###
        p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=2)
        p_e.fit(epochs)
        p_eobject = p_e.data_
        datap_e2 = p_e._reduce_to(reduction_func, target=target, picks=None)
        
        ### Permuttion entropy ###
        p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=4)
        p_e.fit(epochs)
        p_eobject = p_e.data_
        datap_e4 = p_e._reduce_to(reduction_func, target=target, picks=None)
        
        ### Permuttion entropy theta###
        p_e = PermutationEntropy(tmin=tmin, tmax=tmax, kernel=3, tau=8)
        p_e.fit(epochs)
        p_eobject = p_e.data_
        datap_e8 = p_e._reduce_to(reduction_func, target=target, picks=None)

        # =============================================================================
        # wSMI MARKERS
        # =============================================================================
        reduction_func= [{'axis': 'channels_y', 'function': np.median},
             {'axis': 'channels', 'function': np.mean},
             {'axis': 'epochs', 'function': trim_mean80}]
        
        ###wSMI ###
        wSMI = SymbolicMutualInformation(tmin=tmin, tmax=tmax, kernel=3, tau=1, backend="openmp",
                     method_params=None, method='weighted', comment='default')
        wSMI.fit(epochs)
        wSMIobject = wSMI.data_
        datawSMI1 = wSMI._reduce_to(reduction_func, target=target, picks=None)
        
        ###wSMI ###
        wSMI = SymbolicMutualInformation(tmin=tmin, tmax=tmax, kernel=3, tau=2, backend="openmp",
                     method_params=None, method='weighted', comment='default')
        wSMI.fit(epochs)
        wSMIobject = wSMI.data_
        datawSMI2 = wSMI._reduce_to(reduction_func, target=target, picks=None)
        
        ###wSMI ###
        wSMI = SymbolicMutualInformation(tmin=tmin, tmax=tmax, kernel=3, tau=4, backend="openmp",
                     method_params=None, method='weighted', comment='default')
        wSMI.fit(epochs)
        wSMIobject = wSMI.data_
        datawSMI4 = wSMI._reduce_to(reduction_func, target=target, picks=None)
        
        ###wSMI ###
        wSMI = SymbolicMutualInformation(tmin=tmin, tmax=tmax, kernel=3, tau=8, backend="openmp",
                     method_params=None, method='weighted', comment='default')
        wSMI.fit(epochs)
        wSMIobject = wSMI.data_
        datawSMI8 = wSMI._reduce_to(reduction_func, target=target, picks=None)
        
        
        # =============================================================================
        # EVOKED MARKERS
        # =============================================================================
        
        ###Contingent Negative Variation (CNV)###
        cnv = ContingentNegativeVariation(tmin=-0.004, tmax=0.596)
        
        reduction_func = [{'axis': 'epochs', 'function': trim_mean80},
             {'axis': 'channels', 'function': np.mean}]
        
        cnv.fit(epochs_erp)
        cnv_chs= ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']
        roi_cnv = np.array(mne.pick_channels(epochs_erp.info['ch_names'], include=cnv_chs))
        dataCNV = cnv._reduce_to(reduction_func, target=target, picks={
        'epochs': None,
        'channels': roi_cnv})
        
        ###P1###
        reduction_func = [{'axis': 'epochs', 'function': trim_mean80},
         {'axis': 'channels', 'function': np.mean},
         {'axis': 'times', 'function': np.mean}]
        p1 = TimeLockedTopography(tmin=0.068, tmax=0.116, comment='p1')
        p1.fit(epochs_erp)
        p1_chs= ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']
        roi_p1 = np.array(mne.pick_channels(epochs_erp.info['ch_names'], include=p1_chs))
        dataP1 = p1._reduce_to(reduction_func, target=target, picks={
        'epochs': None,
        'channels': roi_p1,
        'times':None})
        
        ###P3a###
        p3a = TimeLockedTopography(tmin=0.28, tmax=0.34, comment='p3a')
        reduction_func = [{'axis': 'epochs', 'function': trim_mean80},
         {'axis': 'channels', 'function': np.mean},
         {'axis': 'times', 'function': np.mean}]
        p3a.fit(epochs_erp)
        p3a_chs= ['AF3', 'AFz', 'AF4', 'F1', 'Fz', 'F2', 'FC1', 'FCz', 'FC2']
        roi_p3a = np.array(mne.pick_channels(epochs_er[.info['ch_names'], include=p3a_chs))
        dataP3a= p3a._reduce_to(reduction_func, target=target, picks={
        'epochs': None,
        'channels': roi_p3a,
        'times':None})
        
        ###P3b###
        p3b = TimeLockedTopography(tmin=0.4, tmax=0.6, comment='p3b')
        reduction_func = [{'axis': 'epochs', 'function': trim_mean80},
         {'axis': 'channels', 'function': np.mean},
         {'axis': 'times', 'function': np.mean}]
        p3b.fit(epochs_erp)
        p3b_chs= ['FC1', 'FCz', 'FC2', 'C1', 'Cz','C2', 'CP1', 'CPz', 'CP2']
        roi_p3b = np.array(mne.pick_channels(epochs_erp.info['ch_names'], include=p3b_chs))
        dataP3b= p3b._reduce_to(reduction_func, target=target, picks={
        'epochs': None,
        'channels': roi_p3b,
        'times':None})
        
        ###Dictionary with all the markers###
        return {'wSMI_1':datawSMI1,'wSMI_2':datawSMI2,'wSMI_4':datawSMI4,'wSMI_8':datawSMI8, 'p_e_1':datap_e1,'p_e_2':datap_e2,'p_e_4':datap_e4,'p_e_8':datap_e8, 'k':datakomplexity, 'b':databeta,'b_n':databetaa_n, 'g':datagamma, 'g_n':datagamma_n, 't':datatheta,'t_n': datatheta_n , 'd':datadelta,
        'd_n':datadelta_n, 'a_n':dataalpha_n, 'a':dataalpha, 'CNV':dataCNV, 'P1':dataP1, 'P3a': dataP3a, 'P3b': dataP3b}



epoch_type = 'evoked'
# epoch_type = 'pseudo-rs'


all_participants = ['VP07','VP08','VP09', 'VP10','VP11','VP12','VP13','VP14','VP18','VP19','VP20','VP22','VP23','VP24','VP25','VP26','VP27','VP28','VP29','VP30','VP31','VP32','VP33','VP35','VP36','VP37']


path = '/media/nicolas.bruno/63f8a366-34b7-4896-a7ce-b5fb4ee78535/Nico/MW_eeg_data/minmarker/' #icm-linux
# path = '/Users/nicobruno/ownCloud/MW_eeg_data/minmarker/' #nico-mac


target = 'epochs'

for i in all_participants:
    participant = i
    
    print('')
    print('#########################################')
    print(f'Computing markers for participant {participant}'
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
    
    df_subtracted = pd.DataFrame.from_dict(all_markers(epochs_subtracted, 0, 0.6, target))
    
    df_subtracted = df_subtracted.assign(
    events = epochs_subtracted.events[:,2],
    label = lambda df: df.events.apply(lambda x: make_str_label(x)).str.split('/'), 
    probe = lambda df: df.label.apply(lambda x: x[0]),
    mind = lambda df: df.label.apply(lambda x: x[1]),
    stimuli = lambda df: df.label.apply(lambda x: x[2]),
    correct = lambda df: df.label.apply(lambda x: x[3]), 
    prev_trial = lambda df: df.label.apply(lambda x: x[4]),
    segment = lambda df: df.label.apply(lambda x: x[5]),
    preproc = 'subtracted',
    epoch_type = epoch_type
    )
    
    df_markers = df_markers.append(df_subtracted)
        
    ##################
    #### With ERP ####
    ##################
    epochs_erp = mne.read_epochs(folder +  participant + '_' + epoch_type + '_' +  'ar_rereferenced_epo.fif')
    epochs_erp.info['description'] = 'biosemi/64' #necessary for wSMI
    epochs_erp =  epochs_erp.pick_types(eeg = True) #EOGs break everything
    
    df_erp = pd.DataFrame.from_dict(all_markers(epochs_erp, 0, 0.6, target))
    
    df_erp = df_erp.assign(
    events = epochs_erp.events[:,2],
    label = lambda df: df.events.apply(lambda x: make_str_label(x)).str.split('/'), 
    probe = lambda df: df.label.apply(lambda x: x[0]),
    mind = lambda df: df.label.apply(lambda x: x[1]),
    stimuli = lambda df: df.label.apply(lambda x: x[2]),
    correct = lambda df: df.label.apply(lambda x: x[3]), 
    prev_trial = lambda df: df.label.apply(lambda x: x[4]),
    segment = lambda df: df.label.apply(lambda x: x[5]),
    preproc = 'erp',
    epoch_type = epoch_type
    )
    df_markers = df_markers.append(df_erp)
    
    
    
    
    df_markers.to_csv(folder+ participant + '_' + epoch_type + '_all_marker.csv')

