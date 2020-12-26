import os
import sys; sys.path.append("..")
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from wyrm.io import convert_mushu_data
from wyrm.processing import segment_dat, append_epo

from utils.options import opt
from utils import utils


## Global Prameters
channels = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6', 
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4', 
                'FFC6', 'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7', 
                'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'
                , 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8', 
                'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6',
                'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1', 'Oz', 'O2', 'OI1', 'OI2', 
                'I1', 'I2']
md = {'class 1': '1','class 2': '2'}


def feature_transform(final_epoch, func=utils.bandpowers):
    dictionary = []
    for i in range(len(final_epoch.axes[0])):
        segment = final_epoch.data[i]
        segment = np.array(segment)
        segment = np.transpose(segment) # 118x50
        features = func(segment)

        dictionary.append(features)
    dictionary = np.array(dictionary)
    return dictionary


def process(filename):
    data_dir = os.path.join("../Datasets/", filename)
    data_path = os.path.join(data_dir, filename + '_cnt.txt')
    label_path = os.path.join(data_dir, filename + '_mrk.txt')

    data_df = pd.read_table(data_path, header=None)
    label_df = pd.read_table(label_path, header=None)

    ## data overview
    print("data shape", data_df.shape)
    print("label shape", label_df.shape)


    ## data 
    label_array = label_df.dropna().values
    train_markers = []
    for events in label_array:
        if events[1] != 0:
            for i in range(0, 400, 50):
                train_markers.append((float(events[0]) + i, str(int(events[1]))))


    markers_subject1_class_1 = [(float(events[0]),str(int(events[1]))) for events in train_markers if events[1]== '1']
    markers_subject1_class_2 = [(float(events[0]),str(int(events[1]))) for events in train_markers if events[1]== '2']


    data_array = data_df.values
    cnt1 = convert_mushu_data(data_array, markers_subject1_class_1, 50, channels)
    cnt2 = convert_mushu_data(data_array, markers_subject1_class_2, 50, channels)

    epoch_subject1_class1 = segment_dat(cnt1, md, [0, 1000]) # 640x50x118
    epoch_subject1_class2 = segment_dat(cnt2, md, [0, 1000]) # 704x50x118
    final_epoch = append_epo(epoch_subject1_class1,epoch_subject1_class2) #1344x50x118
    targets = final_epoch.axes[0]

    methods = ['_csp', '_bandpowers', '_dct', '_wavelet']
    for i, func in enumerate(['_csp', utils.bandpowers, utils.dct_features, utils.wavelet_features]):
        if func == '_csp':
            from mne.decoding import CSP
            csp = CSP(n_components=50, reg=None, log=True, norm_trace=True)
            dictionary = csp.fit_transform(final_epoch.data, targets)
        else:
            dictionary = feature_transform(final_epoch, func)
        

        ## save the data
        res = np.concatenate([dictionary, targets.reshape(-1, 1)], axis=1)
        res_df = pd.DataFrame(res)
        save_path = os.path.join(data_dir, filename + methods[i] + '.csv')
        res_df.to_csv(save_path, index=False)
        print("==> saved data at {}".format(save_path))

    
    ##csp method 



if __name__ == "__main__":
    if opt.data_name != 'all':
        filename = 'data_set_IVa_{}'.format(opt.data_name)
        process(filename)
    else:
        for data_name in ['aa', 'al', 'av', 'aw', 'ay']:
            filename = 'data_set_IVa_{}'.format(data_name)
            process(filename)