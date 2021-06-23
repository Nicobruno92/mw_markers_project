import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    cross_val_score,
    GroupShuffleSplit,
    permutation_test_score,
    StratifiedKFold,
    KFold
)

from sklearn.decomposition import PCA

from sklearn.ensemble import ExtraTreesClassifier


from nice.markers import (KolmogorovComplexity, TimeLockedContrast, PowerSpectralDensityEstimator, PowerSpectralDensitySummary,
                          PowerSpectralDensity, SymbolicMutualInformation, PermutationEntropy, TimeLockedTopography, ContingentNegativeVariation)
import pycsd



def make_str_label(label):
    """
    Makes the string label. Detailing the condtion for the key of the dictionary
    label: numeric label in a predefined format. See table
    """
    label = str(label)
    str_label = ''

    if label[0] == '1':
        str_label += 'PC'
    elif label[0] == '2':
        str_label += 'SC'
    str_label += '/'

    if label[1] == '1':
        str_label += 'on-task'
    elif label[1] == '2':
        str_label += 'about-task'
    elif label[1] == '3':
        str_label += 'distracted'
    elif label[1] == '4':
        str_label += 'dMW'
    elif label[1] == '5':
        str_label += 'sMW'
    str_label += '/'

    if label[2] == '1':
        str_label += 'go'
    elif label[2] == '0':
        str_label += 'nogo'
    str_label += '/'

    if label[3] == '1':
        str_label += 'correct'
    elif label[3] == '0':
        str_label += 'incorrect'
    str_label += '/'
    
    #trial num
    str_label += label[4]
    str_label += '/'
    
    #segment num
    str_label += f's{label[5:]}'


    return (str_label)

def make_correct_labels(raw, nb_prev_trials=5):
    """
    Takes a raw file with raw labels, and returns an array with the markers of interest and the correct labels
    :param raw: a mne raw file
    :param nb_prev_trials: the number of trials to consider before the probe
    :return: array of events, dictionary with the labels of the events
    """
    # read events
    events_from_annot, event_dict = mne.events_from_annotations(raw)

    # fraction events label in different kinds
    SC_labels = [event_dict[your_key] for your_key in [format(x, '02d') for x in range(150, 199)] if
                 your_key in event_dict]
    PC_labels = [event_dict[your_key] for your_key in [format(x, '02d') for x in range(50, 99)] if
                 your_key in event_dict]
    go_response = [event_dict[your_key] for your_key in ['11'] if your_key in event_dict]
    go_trials = [event_dict[your_key] for your_key in ['101', '102', '103', '104', '105', '107', '108', '109'] if
                 your_key in event_dict]
    nogo_trials = [event_dict[your_key] for your_key in ['106'] if your_key in event_dict]
    probe_report = [event_dict[your_key] for your_key in [format(x, 'd') for x in range(1, 6)] if
                    your_key in event_dict]

    # reduce the events to events of interest
    events_labels = SC_labels + PC_labels + go_response + go_trials + nogo_trials + probe_report
    events_of_interest = mne.pick_events(events_from_annot, include=events_labels)

    # convert array of events to pandas for manipulation
    df_events = pd.DataFrame(events_of_interest, columns=['time', '0', 'trigger'])

    df_trials = df_events[df_events['trigger'].isin(go_trials)]

    # format dataframe
    probes_index = []
    for index, row in df_events.iterrows():
        if row.trigger in SC_labels + PC_labels:
            probes_index.append(index)

    df_events['segment'] = 0

    index_m1 = 0
    nb_index = 0
    for index in probes_index:
        df_events.segment[index_m1:index + 2] = nb_index
        index_m1 = index + 2
        nb_index += 1

    df_events['prev_trial'] = 0
    df_events['probe'] = 0
    df_events['mind'] = 0

    index_m1 = 0
    nb_index = 0
    for index in probes_index:
        trial_nb = 1

        for i in range(index - index_m1):
            if df_events.trigger[index] in PC_labels:
                df_events.probe[index - i] = 1
            elif df_events.trigger[index] in SC_labels:
                df_events.probe[index - i] = 2

            #the for loop is beause sometimes it has events between the probe event and the report
            for j in range(5):
                try:
                    if df_events.trigger[index + j] == event_dict['1']:
                        df_events.mind[index - i] = 1
                        break
                    elif df_events.trigger[index + j] == event_dict['2']:
                        df_events.mind[index - i] = 2
                        break
                    elif df_events.trigger[index + j] == event_dict['3']:
                        df_events.mind[index - i] = 3
                        break
                    elif df_events.trigger[index + j] == event_dict['4']:
                        df_events.mind[index - i] = 4
                        break
                    elif df_events.trigger[index + j] == event_dict['5']:
                        df_events.mind[index - i] = 5
                        break
                except:
                    if df_events.trigger[index + j] == event_dict['2']:
                        df_events.mind[index - i] = 2
                        break
                    elif df_events.trigger[index + j] == event_dict['3']:
                        df_events.mind[index - i] = 3
                        break
                    elif df_events.trigger[index + j] == event_dict['4']:
                        df_events.mind[index - i] = 4
                        break
                    elif df_events.trigger[index + j] == event_dict['5']:
                        df_events.mind[index - i] = 5
                        break

            if df_events.trigger[index - i] in go_trials + nogo_trials:

                df_events.prev_trial[index - i] = trial_nb
                trial_nb += 1
            elif df_events.trigger[index - i] in go_response:
                df_events.prev_trial[index - i] = trial_nb

        index_m1 = index + 1

    def case_when_responses(row):
        if row.trigger in go_trials and row.len_segment > 1:
            return (1)
        elif row.trigger in go_trials and row.len_segment == 1:
            return (0)
        elif row.trigger in nogo_trials and row.len_segment > 1:
            return (0)
        elif row.trigger in nogo_trials and row.len_segment == 1:
            return (1)
    
    def make_num_labels(row):
        label = 00000000
        if row.probe == 1:
            label += 10000000
        elif row.probe == 2:
            label += 20000000

        if row.mind == 1:
            label += 1000000
        elif row.mind == 2:
            label += 2000000
        elif row.mind == 3:
            label += 3000000
        elif row.mind == 4:
            label += 4000000
        elif row.mind == 5:
            label += 5000000

        if row.trigger in go_trials:
            label += 100000
        elif row.trigger in nogo_trials:
            label += 000000

        if row.correct == 1:
            label += 10000
        elif row.correct == 0:
            label += 00000

        label += int(row.prev_trial*1000)

        label += int(row.segment)

        return (int(label))
    

    df_events = (df_events
        .assign(
        len_segment=lambda df: df.groupby(['segment', 'prev_trial']).trigger.transform(np.size),
        correct=lambda df: df.apply(case_when_responses, axis=1),
        label=lambda df: df.apply(make_num_labels, axis=1))
    )

    df_events = df_events[df_events.prev_trial <= nb_prev_trials]
    df_events = df_events[df_events.correct.notnull()]
    df_events = df_events[df_events.probe != 0]

    # create dictionary of events names
    dict_final_events = {}

    for index, row in df_events.iterrows():
        str_label = make_str_label(row.label)
        if str_label not in dict_final_events.keys():
            dict_final_events[str_label] = row.label.astype(int)

    # create array of events
    array_events = df_events[['time', '0', 'label']].to_numpy()
    array_events = array_events.astype(int)

    return array_events, dict_final_events


#####EPOCHING#####
def create_epochs(epoch_type, raw, events, event_id, save = False):
    """
    Creates the epochs for the Evoked epochs and the pseudo resting state epochs.
    """
    
    if epoch_type == 'evoked':
    
        tmin, tmax = -0.2, 0.6

        baseline = (None,0)
        
    elif epoch_type == 'pseudo-rs':
        tmin, tmax = -0.8, 0

        baseline = (None,-0.6)
    
    
    epochs = mne.Epochs(raw, events = events, event_id= event_id, baseline = baseline ,
                            tmin = tmin, tmax = tmax, picks = ('eeg', 'eog'), preload = True,
                            verbose = False)
    
    if save == True:
        epochs.save(folder + participant + epoch_type + '_epo.fif', overwrite = True)
    
    return epochs



def set_montage(raw):
    """
    Set the correct montage for the system. It sets the electrodes to EOG. 
    Removes channels that are not used. And fix the channel names for the montage for some recordings
    """
    try:
        raw.set_channel_types({'EXG3': 'eog', 'EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog','EXG1': 'eog', 'EXG2': 'eog', 'EXG7': 'eog', 'EXG8' : 'eog'})

        raw = raw.pick_types(eeg = True, eog = True, exclude = ['EXG1', 'EXG2', 'EXG7', 'EXG8'])

        raw = raw.set_montage('biosemi64') # ask if biosemi or 10-20 montage
        
    except:
        corrected_chs = {'C1-0': 'C1', 'C3-0': 'C3', 'C5-0':'C5', 'C2-0': 'C2', 'C4-0':'C4', 'C6-0':'C6','C1-1':'EXG1',
         'C2-1':'EXG2',
         'C3-1':'EXG3',
         'C4-1':'EXG4',
         'C5-1':'EXG5',
         'C6-1':'EXG6',
         'C7':'EXG7',
         'C8' :'EXG8'}

        raw = raw.rename_channels(corrected_chs)
        
        raw.set_channel_types({'EXG3': 'eog', 'EXG4': 'eog', 'EXG5': 'eog','EXG6': 'eog','EXG1': 'eog', 'EXG2': 'eog', 'EXG7': 'eog', 'EXG8' : 'eog'})

        raw = raw.pick_types(eeg = True, eog = True, exclude = ['EXG1', 'EXG2', 'EXG7', 'EXG8'])

        raw = raw.set_montage('biosemi64') # ask if biosemi or 10-20 montage
    
    return raw


def balance_sample(df, subject, group_var, levels = 2):
    """
    Delete all subjects that don't have an observation for each grouping variable
    df: datafrane
    subject: subject variable
    group_var: grouping variable
    levels: the levels of the grouping variable
    """
    df = df[
        df[subject].isin(
            [
                sid
                for sid, group in df.groupby(subject)
                if len(set(group[group_var])) == levels
            ]
        )
    ]
    return df

def all_markers(epochs, tmin, tmax, target, epochs_erp = None):
        """
        Computes all ther markers for given epochs.
        epochs: the epochs from which to compute the markers
        tmin: min time for computing markers 
        tmax: max time to compute markers
        target: reduction target, epochs or topography
        epochs_erp: if a different preprocessed epochs are used for computing ERP
        
        Evoked markers have already defined times
        """
        if epochs_erp ==None:
            epochs_erp = epochs
        
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
        
        
        
        #Spectral Entropy
        se = PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                         normalize=False, comment='summary_se')
        se.fit(epochs)
        datase = se._reduce_to(reduction_func, target=target, picks=None)
        
        
        #### Spectral Summary ####
        
        reduction_func= [{'axis': 'channels', 'function': np.mean},
             {'axis': 'epochs', 'function': trim_mean80}]
        
        # msf
        msf = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.5, comment='summary_msf')
        msf.fit(epochs)
        datamsf = msf._reduce_to(reduction_func, target=target, picks=None)
        
        #sef90
        sef90 = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.9, comment='summary_sef90')
        sef90.fit(epochs)
        datasef90 = sef90._reduce_to(reduction_func, target=target, picks=None)
        
        #sef95
        sef95 = PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.95, comment='summary_sef95')
        sef95.fit(epochs)
        datasef95 = sef95._reduce_to(reduction_func, target=target, picks=None)

        # =============================================================================
        # INFORMATION THEORY MARKERS
        # =============================================================================
        
        ### Kolgomorov complexity ###
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
        roi_p3a = np.array(mne.pick_channels(epochs_erp.info['ch_names'], include=p3a_chs))
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
        return {'wSMI_1':datawSMI1,'wSMI_2':datawSMI2,'wSMI_4':datawSMI4,'wSMI_8':datawSMI8, 
                'p_e_1':datap_e1,'p_e_2':datap_e2,'p_e_4':datap_e4,'p_e_8':datap_e8, 
                'k':datakomplexity, 'se':datase,
                'msf': datamsf, 'sef90':datasef90, 'sef95':datasef95,
                'b':databeta,'b_n':databetaa_n, 'g':datagamma, 'g_n':datagamma_n, 
                't':datatheta,'t_n': datatheta_n , 'd':datadelta, 'd_n':datadelta_n, 
                'a_n':dataalpha_n, 'a':dataalpha, 
                'CNV':dataCNV, 'P1':dataP1, 'P3a': dataP3a, 'P3b': dataP3b}



def multivariate_classifier(
    data, label, features, model ,pca = False, n_components = 3,  cv_splits = 5,grid_search=False,plot = True, permutation=False, n_permutations = 1000 
):
    """
    data: dataframe with features and labels
    label: name of the column with the labels for the classification
    features: feaure or list of features corresponding to the columns of the data frame with the markers
    model: type of classifier model {SVM or forest}
    pca: if use pca as reduction
    n_components: number of components of the pca
    cv_splits: number of crossvalidations splits
    grid_search: if true it will apply grid search 5cv to find the best parameters of C and gamma, only for the SVM. Deafault: False
    n_permutations: number of permutations
    """
    #pipeline steps init
    steps = [("scaler", StandardScaler())]
    
    if pca == True:
        steps.append(('pca', PCA(n_components = n_components)))
        
    C= 0.001
    gamma=0.1
    
    y, lbl = pd.factorize(data[label])
    X = data[features].astype("float32").values
        
    if grid_search == True:
 

        steps_grid = steps.copy()
        steps_grid.append(("SVM", SVC(probability=True)))
        pipe_grid = Pipeline(steps_grid)

        parameteres = {
            "SVM__C": [0.001, 0.1, 10, 100, 10e5],
            "SVM__gamma": [1,0.5, 0.1, 0.01, 0.001],
        }
        
        grid = GridSearchCV(pipe_grid, param_grid=parameteres, cv=cv_splits, n_jobs=-1)
        grid.fit(X, y)
        
        print(grid.best_params_)
        
        C=grid.best_params_["SVM__C"]
        gamma=grid.best_params_["SVM__gamma"]
        
    if model == 'SVM':
        steps.append(('SVM', SVC(C = C, gamma = gamma, probability = True)))       

        pipe_cv = Pipeline(steps)

        cv = KFold(cv_splits, shuffle=True, random_state = 42)

        aucs = cross_val_score(
            X=X,
            y=y,
            estimator=pipe_cv,
            scoring="roc_auc",
            cv=cv,
        )

  
    if model == 'forest':
        n_estimators = 1000
        steps.append(('Forest',ExtraTreesClassifier(
                n_estimators=n_estimators, max_features='auto', criterion='entropy',
                max_depth=None, random_state=42, class_weight=None)))
        
        pipe_cv = Pipeline(steps)
        
        cv = KFold(cv_splits, shuffle=True, random_state = 42)


        aucs = cross_val_score(
            X=X, y=y, estimator=pipe_cv,
            scoring='roc_auc', cv=cv, groups=np.arange(len(X)), n_jobs = -1)
        
    
    df_auc = pd.DataFrame(aucs, columns=["auc"])
    
    
    if plot == True:
        sns.catplot(x = 'auc', orient = 'h', data = df_auc, kind = 'violin')
        plt.title(f'Mean = {np.mean(df_auc.auc)}; SD = {np.std(df_auc.auc)}')
        plt.axvline(x = 0.5, linestyle = 'dashed')
        plt.show()
        
     
    # Feature importance
    if model == 'forest':
        pipe_cv.fit(X, y)
        variable_importance = pipe_cv.steps[-1][-1].feature_importances_
        sorter = variable_importance.argsort()

        feat_import = pd.DataFrame(np.array([features,variable_importance]).T, 
                                   columns = ['features', 'value']).sort_values('value', ascending = False)
        
        sns.scatterplot(x = feat_import.value, y =feat_import.features)
        plt.title(f'AUC = {np.mean(df_auc.auc)}')
        plt.show()
            
    if permutation == True:
        score, perm_scores, pvalue = permutation_test_score(
            pipe_cv, X, y, scoring="roc_auc", cv=cv, n_permutations=n_permutations, random_state = 42, n_jobs= -1
        )
            

        print(f"p_value = {pvalue}")
        
        plt.hist(perm_scores, bins=20, density=True)
        plt.axvline(score, ls="--", color="r")
        score_label = (
            f"Score on original\ndata: {score:.2f}\n" f"(p-value: {pvalue:.3f})"
        )
        plt.text(score, np.max(perm_scores), score_label, fontsize=12)
        plt.xlabel("Accuracy score")
        plt.ylabel("Probability")
        plt.show()
        
    

    return df_auc

def univariate_classifier(
    data, label, feature, model, grid_search=False, permutation=False, n_permutations = 1000, perm_plot = False
):
    """
    data: dataframe with features and labels
    label: name of the column with the labels for the classification
    features: feaure or list of features corresponding to the columns of the data frame with the markers
    model: type of classifier model
    grid_search: if true it will apply grid search 5cv to find the best parameters of C and gamma, only for the SVM. Deafault: False
    """
    y, lbl = pd.factorize(data[label])
    X = data[feature].astype("float32").values.reshape(-1,1)
    
    if model == "SVM":
        if grid_search == True:

            steps = [("scaler", StandardScaler()), ("SVM", SVC(probability=True))]
            pipe = Pipeline(steps)

            parameteres = {
                "SVM__C": [0.001, 0.1, 10, 100, 10e5],
                "SVM__gamma": [0.1, 0.01],
            }
            grid = GridSearchCV(pipe, param_grid=parameteres, cv=5, n_jobs=-1)
            grid.fit(X, y)

            steps = [
                ("scaler", StandardScaler()),
                (
                    "SVM",
                    SVC(
                        C=grid.best_params_["SVM__C"],
                        gamma=grid.best_params_["SVM__gamma"],
                        probability=True,
                    ),
                ),
            ]
            
            pipe_cv = Pipeline(steps)

            cv = StratifiedKFold(10, shuffle=True, random_state = 42)

            aucs = cross_val_score(
                X=X,
                y=y,
                estimator=pipe_cv,
                scoring="roc_auc",
                cv=cv,
            )
            print(grid.best_params_)

        else:


            steps = [
                ("scaler", StandardScaler()),
                ("SVM", SVC(C=0.001, gamma=0.1, kernel="rbf", probability=True)),
            ]
            pipe_cv = Pipeline(steps)

            cv = StratifiedKFold(10, shuffle=True, random_state = 42)

            aucs = cross_val_score(
                X=X,
                y=y,
                estimator=pipe_cv,
                scoring="roc_auc",
                cv=cv,
            )
            

    if model == 'forest':
            steps = [
            ("scaler", StandardScaler()),
                       ]
            n_estimators = 1000
            steps.append(('Forest',ExtraTreesClassifier(
                n_estimators=n_estimators, max_features='auto', criterion='entropy',
                max_depth=None, random_state=42, class_weight=None)))
            pipe_cv = Pipeline(steps)

            cv = StratifiedKFold(10, shuffle=True, random_state = 42)

            aucs = cross_val_score(
                X=X,
                y=y,
                estimator=pipe_cv,
                scoring="roc_auc",
                cv=cv,
            )

    print(f'AUC {feature} = {np.mean(aucs)}')
            
    if permutation == True:
        score, perm_scores, pvalue = permutation_test_score(
            pipe_cv, X, y, scoring="roc_auc", cv=cv, n_permutations=n_permutations, n_jobs =-1
        )
            

        print(f"p_value = {pvalue}")
        
        
        if perm_plot == True:

            plt.hist(perm_scores, bins=20, density=True)
            plt.axvline(score, ls="--", color="r")
            score_label = (
                f"Score on original\ndata: {score:.2f}\n" f"(p-value: {pvalue:.3f})"
            )
            plt.text(score, np.max(perm_scores), score_label, fontsize=12)
            plt.xlabel("Accuracy score")
            plt.ylabel("Probability")
            plt.show()
        
        return aucs, pvalue
    
    else:
        return aucs, 0
    

def bad_participant(epochs, probe, mind):
    df = epochs.to_data_frame()
    df = (df.assign(
        label = lambda df: df.condition.apply(lambda x: x.split('/')), 
        probe = lambda df: df.label.apply(lambda x: x[0]),
        mind = lambda df: df.label.apply(lambda x: x[1]),
        stimuli = lambda df: df.label.apply(lambda x: x[2]),
        correct = lambda df: df.label.apply(lambda x: x[3]), 
        prev_trial = lambda df: df.label.apply(lambda x: int(x[4])),
        segment = lambda df: df.label.apply(lambda x: x[5]),
        )
        .query("mind in ['on-task','dMW', 'sMW']")
        .query("stimuli == 'go'")
        .query("correct == 'correct'")
        .query('prev_trial <= 5')
        .assign(
        mind2 = lambda df: np.where(df.mind == 'on-task', 'on-task', 'mw'))
        .groupby(['prev_trial', 'segment']).first())
    
    df = df[df['probe']==probe].groupby([mind]).filter(lambda x: len(x) >= 8) #min nbr of trials for condition
    
    return len(set(df[mind])) != 2