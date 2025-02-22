# -*- coding: utf-8 -*-


import os
import numpy as np
import mne
from sklearn.preprocessing import normalize
import pickle 

#define the path where the list with the recordings is stored
path = '/home/u956278/EEG_syntethic_data/data/EEG_raw/'

#create a list with the useless channels as said in the documentation provided ("Load_cnt_file_with_mne.ipynb")
useless_ch = ['M1', 'M2', 'VEO', 'HEO']

#create a list with the file names of each recording as strings in "subjectID_sessionID_date.cnt" format
eeg_raw_list = [f for f in os.listdir(path) if f.endswith('.cnt')]

#create lists for train and test set
list_happy_train, list_fear_train, list_neutral_train, list_sad_train, list_disgust_train = [], [], [], [], []
list_happy_test, list_fear_test, list_neutral_test, list_sad_test, list_disgust_test = [], [], [], [], []

#look at the start and end times for each emotion in each session
"""Session 1:
start_second: [30,  132, 287, 555, 773, 982,  1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
end_second:   [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
              Happy	Fear Neutral	Sad	Disgust	Happy	Fear	Neutral	Sad	Disgust	Happy	Fear	Neutral	Sad	Disgust
Session 2:
start_second: [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741]
end_second:  [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
           Sad Fear Neutral Disgust Happy	Happy	Disgust	Neutral	Sad	Fear Neutral Happy Fear Sad	Disgust
Session 3:
    start_second: [30, 353, 478, 674, 825, 908,  1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
    end_second:   [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
                  Sad Fear Neutral Disgust Happy Happy Disgust Neutral Sad Fear Neutral Happy	Fear Sad Disgust"""

#define a function for creating lists of start and end times for each session
def matrix_indices(session):
  session_index = []

  if session == 1:
    #define the labels for each session (done by hand -copypaste from above)
    labels = "Happy	Fear Neutral	Sad	Disgust	Happy	Fear Neutral Sad Disgust Happy	Fear Neutral Sad Disgust".split()
    #define the start and end times for each session (done by hand -copypaste from above)
    start, end = (30,  132, 287, 555, 773, 982,  1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204), (102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359)
  elif session == 2:
    labels = "Sad Fear Neutral Disgust Happy	Happy	Disgust	Neutral	Sad	Fear Neutral Happy Fear Sad	Disgust".split()
    start, end = (30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741), (267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817)
  elif session == 3:
    labels = "Sad Fear Neutral Disgust Happy Happy Disgust Neutral Sad Fear Neutral Happy	Fear Sad Disgust".split()
    start, end = (30, 353, 478, 674, 825, 908,  1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888), (321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066)
  
  #adjust for 1000hz sampling
  for a, b in zip(start, end):
    session_index.append((a*1000, b*1000))
  
  #create a temporary dictionary to store the time intervals as tuples and emotions as keys
  temp_dic = {}
  for a, b in zip(session_index, labels): 
    if temp_dic.get(b, -1) == -1:
      temp_dic[b] = [a]
    else:
      temp_dic[b].append(a)
  
  #we want to sort the recordings so that each matrix keeps the emotions in the same order (of session 1, chosen for ease)
  indexes = []
  for i in "Happy	Fear Neutral	Sad	Disgust".split():
    indexes.append(temp_dic[i])

  return indexes

def process_STFT(STFT):
    img_list = []
    #STFT obj has 3 dims: channels x frequencies x datapoint, let's work on datapoints 
    for i in range(STFT.shape[2]):
        #take the abs val of the ith datapoint cutting up to 100hz (higher freqs are deemed not relevant for brain activity inspection)
        abs_val = np.abs(STFT[:, :100, i])
        #let's normalize the data and take the abs val to make it 2d and append it to the final list 
        data2D = normalize(abs_val, axis=1).T
        img_list.append(data2D)
    return img_list

for eeg in eeg_raw_list:
    #take subject id and session from filename
    subject_id, session, _ = eeg.split('_')
    subject_id, session = int(subject_id), int(session)

    #load the data using mne, drop useless channels and create a matrix 
    eeg_path = os.path.join(path, eeg)
    eeg_raw = mne.io.read_raw_cnt(eeg_path).drop_channels(useless_ch)
    data_matrix = eeg_raw.get_data()

    #create a list of ranges and slice the data_matrix according to the ranges for each emotion
    matrix_ranges = matrix_indices(session)
    
    matrix_ha = [data_matrix[:, start:end] for start, end in matrix_ranges[0]]
    matrix_fear = [data_matrix[:, start:end] for start, end in matrix_ranges[1]]
    matrix_neutral = [data_matrix[:, start:end] for start, end in matrix_ranges[2]]
    matrix_sad = [data_matrix[:, start:end] for start, end in matrix_ranges[3]]
    matrix_disgust = [data_matrix[:, start:end] for start, end in matrix_ranges[4]]

    #perform Short-Time Fourier Transform on each emotion matrix
    STFT_happy = np.concatenate([mne.time_frequency.stft(mat, 1000) for mat in matrix_ha], axis=2)
    STFT_fear = np.concatenate([mne.time_frequency.stft(mat, 1000) for mat in matrix_fear], axis=2)
    STFT_neutral = np.concatenate([mne.time_frequency.stft(mat, 1000) for mat in matrix_neutral], axis=2)
    STFT_sad = np.concatenate([mne.time_frequency.stft(mat, 1000) for mat in matrix_sad], axis=2)
    STFT_disgust = np.concatenate([mne.time_frequency.stft(mat, 1000) for mat in matrix_disgust], axis=2)

    #save 2d imgs into train and test lists
    if subject_id <= 13: #train
        list_happy_train.extend(process_STFT(STFT_happy)) 
        list_fear_train.extend(process_STFT(STFT_fear)) 
        list_neutral_train.extend(process_STFT(STFT_neutral)) 
        list_sad_train.extend(process_STFT(STFT_sad)) 
        list_disgust_train.extend(process_STFT(STFT_disgust))
    else: #test
        list_happy_test.extend(process_STFT(STFT_happy)) 
        list_fear_test.extend(process_STFT(STFT_fear)) 
        list_neutral_test.extend(process_STFT(STFT_neutral)) 
        list_sad_test.extend(process_STFT(STFT_sad)) 
        list_disgust_test.extend(process_STFT(STFT_disgust))
#save train 
train_lists = [list_happy_train, list_fear_train, list_neutral_train, list_sad_train, list_disgust_train]
train_labels = "list_happy_train, list_fear_train, list_neutral_train, list_sad_train, list_disgust_train".split(",")
for element, label in zip(train_lists, train_labels): 
  with open(f'/home/u956278/EEG_syntethic_data/data/stft/EEG_to_pickle_stft/{label}.pickle', 'wb') as handle:
      pickle.dump(element, handle, protocol=pickle.HIGHEST_PROTOCOL)

#save test
test_lists = [list_happy_test, list_fear_test, list_neutral_test, list_sad_test, list_disgust_test]
test_labes = "list_happy_test, list_fear_test, list_neutral_test, list_sad_test, list_disgust_test".split(",")
for element, label in zip(test_lists, test_labes): 
  with open(f'/home/u956278/EEG_syntethic_data/data/stft/EEG_to_pickle_stft/{label}.pickle', 'wb') as handle:
      pickle.dump(element, handle, protocol=pickle.HIGHEST_PROTOCOL)

