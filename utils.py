import mne
import numpy as np
import pandas as pd


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
        label = 00000
        if row.probe == 1:
            label += 10000
        elif row.probe == 2:
            label += 20000

        if row.mind == 1:
            label += 1000
        elif row.mind == 2:
            label += 2000
        elif row.mind == 3:
            label += 3000
        elif row.mind == 4:
            label += 4000
        elif row.mind == 5:
            label += 5000

        if row.trigger in go_trials:
            label += 100
        elif row.trigger in nogo_trials:
            label += 000

        if row.correct == 1:
            label += 10
        elif row.correct == 0:
            label += 00

        label += row.prev_trial

        return (label)

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

        str_label += str(label[4])

        return (str_label)

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

