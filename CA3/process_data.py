import os

import pandas as pd

# Set data folder path
DATA_FOLDER = r'D:/NUS_TERM2_CA3/MAREA_dataset'

ACTIVITY_FOLDER = os.path.join(DATA_FOLDER, 'Activity Timings')
SUBJECT_FOLDER = os.path.join(DATA_FOLDER, 'Subject_Data_txt_format')

# Define Activity Labels
indoor_label = ['tread_flat_walk_start',
                'tread_flat_walk_end',
                'tread_flat_run_end',
                'tread_slope_walk_start',
                'tread_slope_walk_end',
                'indoor_flat_walk_start',
                'indoor_flat_walk_end',
                'indoor_flat_run_end'
                ]

outdoor_label = ['outdoor_walk_start',
                 'outdoor_walk_end',
                 'outdoor_run_end']

indoor_time_df = pd.read_csv(os.path.join(ACTIVITY_FOLDER, 'Indoor Experiment Timings.txt')
                             , names=indoor_label)

outdoor_time_df = pd.read_csv(os.path.join(ACTIVITY_FOLDER, 'Outdoor Experiment Timings.txt')
                              , names=outdoor_label)

indoor_time_df["subject"] = ["Sub" + str(i) for i in range(1, 12)]
outdoor_time_df["subject"] = ["Sub" + str(j) for j in range(12, 21)]

print(indoor_time_df)
print(outdoor_time_df)

pos_list = ['LF', 'RF', 'Waist', 'Wrist']
sub_list = ["Sub" + str(i) for i in range(1, 21)]
sub_list.remove('Sub4')

from scipy import sparse
from scipy.sparse.linalg import spsolve
import numpy as np
import pywt


def alsbase(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def denoise(signal_orig):
    coeffs_orig = pywt.wavedec(signal_orig, 'db4')
    coeffs_filter = coeffs_orig.copy()
    threshold = 0.8
    for i in range(1, len(coeffs_orig)):
        coeffs_filter[i] = pywt.threshold(coeffs_orig[i], threshold * max(coeffs_orig[i]))

    signal_denoised = pywt.waverec(coeffs_filter, 'db4')
    s = pd.DataFrame(signal_denoised)
    return s


new_names = ['accX_LF', 'accY_LF', 'accZ_LF',
             'accX_RF', 'accY_RF', 'accZ_RF',
             'accX_Waist', 'accY_Waist', 'accZ_Waist',
             'accX_Wrist', 'accY_Wrist', 'accZ_Wrist'
             ]

sub_df = None

# sub_list1 = ['Sub12']
for sub in sub_list:
    lf_df = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'LF.txt'))
    rf_df = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'RF.txt'))
    waist_df = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'Waist.txt'))
    wrist_df = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'Wrist.txt'))
    sub_df = pd.concat([lf_df, rf_df, waist_df, wrist_df], axis=1)
    sub_df.columns = new_names
    print(sub_df.head())

    sub_df_new = sub_df.copy()
    sub_df_new = denoise(sub_df_new.values)
    sub_df_new.columns = new_names
    print(sub_df_new.head())

    for column in new_names:
        sub_df_new[column] = sub_df_new[column] - alsbase(sub_df_new[column], 10 ^ 5, 0.000005, niter=10)

    n = int(sub[3:])
    if n > 11:
        sub_row = outdoor_time_df[outdoor_time_df['subject'] == sub]
        tmp = sub_row.iloc[0]
        sub_df_new.loc[0:tmp['outdoor_walk_end'], 'label'] = 'outdoor_walk'
        sub_df_new.loc[tmp['outdoor_walk_end']: tmp['outdoor_run_end'], 'label'] = 'outdoor_run'
    else:
        sub_row = indoor_time_df[indoor_time_df['subject'] == sub]
        tmp = sub_row.iloc[0]
        sub_df_new.loc[0:tmp['tread_flat_walk_end'], 'label'] = 'tread_flat_walk'
        sub_df_new.loc[tmp['tread_flat_walk_end']: tmp['tread_flat_run_end'],
        'label'] = 'tread_flat_run'
        sub_df_new.loc[tmp['tread_flat_run_end']: tmp['tread_slope_walk_start'], 'label'] = 'NA'
        sub_df_new.loc[tmp['tread_slope_walk_start']: tmp['tread_slope_walk_end'],
        'label'] = 'tread_slope_walk'
        sub_df_new.loc[tmp['tread_slope_walk_end']: tmp['indoor_flat_walk_start'], 'label'] = 'NA'
        sub_df_new.loc[tmp['indoor_flat_walk_start']: tmp['indoor_flat_walk_end'],
        'label'] = 'indoor_flat_walk'
        sub_df_new.loc[tmp['indoor_flat_walk_end']: tmp['indoor_flat_run_end'],
        'label'] = 'indoor_flat_run'

    print(sub_df_new)
    sub_df_new.to_csv(sub + '_processed.csv')
