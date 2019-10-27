# %% md

# Import Data

# %%

import itertools
import os
# plt.style.use('ggplot')
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.layers import Dense, Dropout, LSTM, Input
from keras.models import Model
from keras.utils import to_categorical
from mlxtend.plotting import plot_confusion_matrix
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM, Activation
# from keras.callbacks import EarlyStopping

filterwarnings('ignore')
# % matplotlib
# inline

# %%

# set data folder path
# DATA_FOLDER = 'D:\\NUS\\semester 2\\Course 3\\CA\\MAREA_dataset'
# DATA_FOLDER = '/Users/jiahao/Downloads/MAREA_dataset'
DATA_FOLDER = r'D:/NUS_TERM2_CA3/MAREA_dataset'
# DATA_FOLDER = 'C:/Users/david/Documents/CA3/MAREA_dataset-201/MAREA_dataset'
ACTIVITY_FOLDER = os.path.join(DATA_FOLDER, 'Activity Timings')
SUBJECT_FOLDER = os.path.join(DATA_FOLDER, 'Subject Data_txt format')
PROCESSED_FOLDER = os.path.join(DATA_FOLDER, 'Processed_data')

# define activity timing labels
label_indoor = ['tread_flat_walk_start',
                'tread_flat_walk_end',
                'tread_flat_run_end',
                'tread_slope_walk_start',
                'tread_slope_walk_end',
                'indoor_flat_walk_start',
                'indoor_flat_walk_end',
                'indoor_flat_run_end']

label_outdoor = ['outdoor_walk_start',
                 'outdoor_walk_end',
                 'outdoor_run_end']

# prepare timing index for different activities
df_indoor_time = pd.read_csv(os.path.join(ACTIVITY_FOLDER, 'Indoor Experiment Timings.txt')
                             , names=label_indoor)

df_outdoor_time = pd.read_csv(os.path.join(ACTIVITY_FOLDER, 'Outdoor Experiment Timings.txt')
                              , names=label_outdoor)

df_indoor_time["subject"] = ["Sub" + str(i) for i in range(1, 12)]
df_outdoor_time["subject"] = ["Sub" + str(j) for j in range(12, 21)]

# %%

# set up activity column names
axis_list = ['accX', 'accY', 'accZ']
pos_list = ['LF', 'RF', 'Waist', 'Wrist']
sub_list = ['Sub' + str(i) for i in range(1, 21)]
column_names = [f"{y}_{x}" for x, y in itertools.product(pos_list, axis_list)]

# TODO: purposely exclude subject 4 first as missing data -- dont know how to deal with missing data for signal
sub_list.remove('Sub4')

# %%

column_names

# %%

# create master dataframe
const_master_df = pd.DataFrame()
for sub in sub_list:
    df_lf = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'LF.txt'))
    df_rf = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'RF.txt'))
    df_waist = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'Waist.txt'))
    df_wrist = pd.read_csv(os.path.join(SUBJECT_FOLDER, sub + '_' + 'Wrist.txt'))
    df_sub = pd.concat([df_lf, df_rf, df_waist, df_wrist], axis=1)
    df_sub.columns = column_names

    df_sub = df_sub.copy()
    n = int(sub[3:])
    if n > 11:
        sub_row = df_outdoor_time[df_outdoor_time['subject'] == sub]
        tmp = sub_row.iloc[0]
        df_sub.loc[0:tmp['outdoor_walk_end'], 'label'] = 'outdoor_walk'
        df_sub.loc[tmp['outdoor_walk_end']: tmp['outdoor_run_end'], 'label'] = 'outdoor_run'
    else:
        sub_row = df_indoor_time[df_indoor_time['subject'] == sub]
        tmp = sub_row.iloc[0]
        df_sub.loc[0:tmp['tread_flat_walk_end'], 'label'] = 'tread_flat_walk'
        df_sub.loc[tmp['tread_flat_walk_end']: tmp['tread_flat_run_end'], 'label'] = 'tread_flat_run'
        df_sub.loc[tmp['tread_flat_run_end']: tmp['tread_slope_walk_start'], 'label'] = 'rest'
        df_sub.loc[tmp['tread_slope_walk_start']: tmp['tread_slope_walk_end'], 'label'] = 'tread_slope_walk'
        df_sub.loc[tmp['tread_slope_walk_end']: tmp['indoor_flat_walk_start'], 'label'] = 'rest'
        df_sub.loc[tmp['indoor_flat_walk_start']: tmp['indoor_flat_walk_end'], 'label'] = 'indoor_flat_walk'
        df_sub.loc[tmp['indoor_flat_walk_end']: tmp['indoor_flat_run_end'], 'label'] = 'indoor_flat_run'

    df_sub['subject'] = sub
    const_master_df = const_master_df.append(df_sub)
    # print(df.shape)

# %%

const_master_df.head(5)


# %%

def PreprocessingSignal(df, label, subject, feature, window,
                        wavelet_args={"type": "Y",
                                      "threshold": 2,
                                      "wavedec_options": {"wavelet": "db4", "level": 2},
                                      "waverec_options": {"wavelet": "db4"}},
                        window_args={"type": "no_overlap"}
                        ):
    df = df.loc[(df['label'] == label) & (df['subject'] == subject)]

    ### Do wavelet transform or NOT ###
    if wavelet_args["type"] == "Y":
        # wavelet_args = {"threshold":2, "options":{"wavelet":"db4", "level":0.8}}
        # Do wavelet transform
        signal_orig = df[feature].values
        args1 = wavelet_args["wavedec_options"]
        coeffs_orig = pywt.wavedec(signal_orig, **args1)
        coeffs_filter = coeffs_orig.copy()
        threshold = wavelet_args["threshold"]
        for i in range(1, len(coeffs_orig)):
            coeffs_filter[i] = pywt.threshold(coeffs_orig[i], threshold * max(coeffs_orig[i]))
        args2 = wavelet_args["waverec_options"]
        signal_denoised = pywt.waverec(coeffs_filter, **args2)
        to_process_df = pd.DataFrame(signal_denoised)
    else:
        tmp_df = df[feature].reset_index()
        to_process_df = tmp_df.drop(columns=["index"])
        to_process_df.columns = [0]
    ### Do wavelet transform or NOT ###

    min_index = min(to_process_df.index)
    max_index = max(to_process_df.index)

    ### Define Method to cut signal into windows ###
    if window_args["type"] == "no_overlap":
        # window_args = {"type":"no_overlap"}
        index_list = range(min_index, max_index + 1, int(window))
    elif window_args["type"] == "with_overlap":
        # window_args = {"type":"with_overlap", "overlap_perc":0.5}
        overlap_perc = window_args["overlap_perc"]
        step = int(window / (1 / overlap_perc))
        index_list = range(min_index, max_index + 1, step)
    elif window_args["type"] == "by_peaks":
        index_list = window_args["peaks_index"]
    ### Define Method to cut signal into windows ###

    ### Cut signal into windows ###
    windowed_selected_chunk_array = []
    for index in index_list:
        windowed_selected_chunk = to_process_df[0].iloc[index:index + window]
        if windowed_selected_chunk.shape[0] == window:
            windowed_selected_chunk_array.append(windowed_selected_chunk.values)
    output_np_arr = np.array(windowed_selected_chunk_array)
    ### Cut signal into windows ###

    output_np_label = np.asarray([label] * output_np_arr.shape[0])

    return output_np_arr, output_np_label


# %%

# feature_list = column_names TODO: is this line still needed?

# Create
const_indoor_activity = ['rest',
                         'tread_flat_walk',
                         'tread_flat_run',
                         'tread_slope_walk',
                         'indoor_flat_walk',
                         'indoor_flat_run'
                         ]
const_outdoor_activity = ['outdoor_walk', 'outdoor_run']

const_indoor_sub = ['Sub' + str(i) for i in range(1, 12)]
const_outdoor_sub = ['Sub' + str(i) for i in range(12, 21)]
const_indoor_sub.remove('Sub4')


# %%

def PrepareFeature(df, feature, split_method, window, wavelet_args, window_args, train_size=0.8):
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()

    # TODO: find_peaks from a given column
    # NOTE: code will break when using -> window_args = {"type":"by_peaks", "peaks_index":peaks_index_list}

    # method 1: split within each subject
    if split_method == 'TrainTestSplitWithinSubject':
        # indoor activity
        for sub in const_indoor_sub:
            for activity in const_indoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list

                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)

                train_len = int(train_size * len(output_np_arr))
                tr_x = output_np_arr[:train_len, :]
                ts_x = output_np_arr[train_len:len(output_np_arr), :]

                test_len = int(train_size * len(output_label))
                tr_y = output_label[:test_len]
                ts_y = output_label[test_len:len(output_label)]

                X_train = X_train.append(pd.DataFrame(tr_x))
                X_test = X_test.append(pd.DataFrame(ts_x))
                y_train = y_train.append(pd.DataFrame(tr_y))
                y_test = y_test.append(pd.DataFrame(ts_y))

                # outdoor activity
        for sub in const_outdoor_sub:
            for activity in const_outdoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list

                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)

                train_len = int(train_size * len(output_np_arr))
                tr_x = output_np_arr[:train_len, :]
                ts_x = output_np_arr[train_len:len(output_np_arr), :]

                test_len = int(train_size * len(output_label))
                tr_y = output_label[:test_len]
                ts_y = output_label[test_len:len(output_label)]

                X_train = X_train.append(pd.DataFrame(tr_x))
                X_test = X_test.append(pd.DataFrame(ts_x))
                y_train = y_train.append(pd.DataFrame(tr_y))
                y_test = y_test.append(pd.DataFrame(ts_y))

    # method 2: split all samples randomly
    if split_method == 'Random':
        #             output = np.asarray(list())
        #             label = np.asarray(list())
        output = np.empty((0, window), float)
        label = np.empty((0,), float)

        # indoor activity
        for sub in const_indoor_sub:
            for activity in const_indoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                output = np.append(output, output_np_arr, axis=0)
                label = np.append(label, output_label, axis=0)

        # outdoor activity:
        for sub in const_outdoor_sub:
            for activity in const_outdoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                output = np.append(output, output_np_arr, axis=0)
                label = np.append(label, output_label, axis=0)

        train_len = int(train_size * len(output))
        tr_x = output[:train_len, :]
        ts_x = output[train_len:len(output), :]

        test_len = int(train_size * len(label))
        tr_y = label[:test_len]
        ts_y = label[test_len:len(label)]

        X_train = X_train.append(pd.DataFrame(tr_x))
        X_test = X_test.append(pd.DataFrame(ts_x))
        y_train = y_train.append(pd.DataFrame(tr_y))
        y_test = y_test.append(pd.DataFrame(ts_y))

    # method 3: only keep several subjects in the train,
    #           put other subjects in the test
    if split_method == 'DifferentSubjectsInTrainTest':
        train_sub = np.empty((0, window), float)
        train_label_sub = np.empty((0,), float)
        test_sub = np.empty((0, window), float)
        test_label_sub = np.empty((0,), float)

        # indoor activity
        indoor_sub_train_len = int(train_size * len(const_indoor_sub))
        indoor_sub_train = const_indoor_sub[:indoor_sub_train_len]
        indoor_sub_test = const_indoor_sub[indoor_sub_train_len:len(const_indoor_sub)]

        for sub in indoor_sub_train:
            for activity in const_indoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    # print("THIS", activity, peaks_index_list)
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                train_sub = np.append(train_sub, output_np_arr, axis=0)
                train_label_sub = np.append(train_label_sub, output_label, axis=0)

        for sub in indoor_sub_test:
            for activity in const_indoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                test_sub = np.append(test_sub, output_np_arr, axis=0)
                test_label_sub = np.append(test_label_sub, output_label, axis=0)

        # outdoor activity
        outdoor_sub_train_len = int(train_size * len(const_outdoor_sub))
        outdoor_sub_train = const_outdoor_sub[:outdoor_sub_train_len]
        outdoor_sub_test = const_outdoor_sub[outdoor_sub_train_len:len(const_outdoor_sub)]

        for sub in outdoor_sub_train:
            for activity in const_outdoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                train_sub = np.append(train_sub, output_np_arr, axis=0)
                train_label_sub = np.append(train_label_sub, output_label, axis=0)

        for sub in outdoor_sub_test:
            for activity in const_outdoor_activity:
                if window_args["type"] == "by_peaks":
                    data_df = df.loc[(df['label'] == activity) & (df['subject'] == sub)]
                    args = window_args["find_peaks_options"]
                    peak_col = feature
                    peaks_index_list = list(find_peaks(data_df[peak_col], **args)[0])
                    window_args["peaks_index"] = peaks_index_list
                output_np_arr, output_label = PreprocessingSignal(df, activity, sub, feature, window, wavelet_args,
                                                                  window_args)
                test_sub = np.append(test_sub, output_np_arr, axis=0)
                test_label_sub = np.append(test_label_sub, output_label, axis=0)

        X_train = pd.DataFrame(train_sub)
        y_train = pd.DataFrame(train_label_sub)
        X_test = pd.DataFrame(test_sub)
        y_test = pd.DataFrame(test_label_sub)

    return X_train, X_test, y_train, y_test


# %%

# SAMPLE #

def prepare_single_feature_to_csv(feature='accY_LF', split_method='TrainTestSplitWithinSubject', window=512,
                                  wavelet_args={"type": "N"},
                                  window_args={"type": "by_peaks", "find_peaks_col": "accX_LF",
                                               "find_peaks_options": {"prominence": 30, "height": 20}}):
    # feature = 'accY_LF'
    # feature_list = column_names
    # for feature in column_names:
    # (example) feature = accX_LF (which is stored in the column_names)

    X_train, X_test, y_train, y_test = PrepareFeature(const_master_df, feature, split_method, window, wavelet_args,
                                                      window_args)

    X_train_filename = "_".join(
        ['X_train', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'
    X_test_filename = "_".join(
        ['X_test', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'
    y_train_filename = "_".join(
        ['y_train', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'
    y_test_filename = "_".join(
        ['y_test', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'

    X_train.to_csv(X_train_filename, header=None, index=False, sep='\t')
    X_test.to_csv(X_test_filename, header=None, index=False, sep='\t')

    pd.DataFrame(y_train).to_csv(y_train_filename, header=None, index=False, sep='\t')
    pd.DataFrame(y_test).to_csv(y_test_filename, header=None, index=False, sep='\t')

    print("Saved X_train to %s" % (X_train_filename))
    print("Saved X_test to %s" % (X_test_filename))
    print("Saved X_train to %s" % (y_train_filename))
    print("Saved X_test to %s" % (y_test_filename))


# %%

def load_file(filepath):
    dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# %%

def load_group(filenames):
    loaded = list()
    for name in filenames:
        data = load_file(name)
        loaded.append(data)

    loaded = np.dstack(loaded)
    return loaded


# %%

def load_dataset(feature_list, window, wavelet_args, window_args, split_method):
    # X_train, y_train = load_dataset_group('train')  # load_group(X_filenames), load_file(y_filename)
    # X_test, y_test = load_dataset_group('test')

    X_train_filenames = []
    X_test_filenames = []

    for feature in [feature_list]:
        X_train_filenames.append("_".join(
            ['X_train', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
             split_method]) + '.txt')

        X_test_filenames.append("_".join(
            ['X_test', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
             split_method]) + '.txt')

    # X_train_filenames = ['X_train_accY_LF_Window512_WaveletN_by_peaks_TrainTestSplitWithinSubject.txt']
    # y_train_filename = 'y_train_accY_LF_Window512_WaveletN_by_peaks_TrainTestSplitWithinSubject.txt'

    y_train_filename = "_".join(
        ['y_train', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'

    y_test_filename = "_".join(
        ['y_test', feature, "Window" + str(window), "Wavelet" + wavelet_args["type"], window_args["type"],
         split_method]) + '.txt'

    X_train = load_group(X_train_filenames)
    y_train = load_file(y_train_filename)

    # X_test_filenames = ['X_test_accY_LF_Window512_WaveletN_by_peaks_TrainTestSplitWithinSubject.txt']
    # y_test_filename = 'y_test_accY_LF_Window512_WaveletN_by_peaks_TrainTestSplitWithinSubject.txt'
    X_test = load_group(X_test_filenames)
    y_test = load_file(y_test_filename)

    activity_to_num_mapping = {
        "rest": 0,
        # indoor
        "tread_flat_walk": 1,
        "tread_flat_run": 2,
        "tread_slope_walk": 3,
        "indoor_flat_walk": 4,
        "indoor_flat_run": 5,
        # outdoor
        "outdoor_walk": 6,
        "outdoor_run": 7
    }

    y_train = np.vectorize(activity_to_num_mapping.get)(y_train)
    y_test = np.vectorize(activity_to_num_mapping.get)(y_test)

    # convert to binary class matrix
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, y_train, X_test, y_test


# %%

# create LSTM Model
def createLSTMModel(n_timesteps, n_features, n_outputs):
    ipt = Input(shape=(n_timesteps, n_features))
    x = LSTM(100)(ipt)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(n_outputs, activation='softmax')(x)

    model = Model(inputs=ipt, outputs=x)

    # Sequential API
    # model = Sequential()
    # model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    # model.add(Dropout(0.5))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(n_outputs, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# %% md

def train_predict_plot(X_train, y_train, X_test, y_test, model_name='v1', verbose=1, epochs=5, batch_size=128):
    model = createLSTMModel(n_timesteps=X_train.shape[1], n_features=X_train.shape[2], n_outputs=y_train.shape[1])
    model.summary()

    filepath = os.path.join(model_name + ".hdf5")
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max')

    # Log the epoch detail into csv
    csv_logger = CSVLogger(os.path.join(model_name + '.csv'))
    callbacks_list = [checkpoint, csv_logger]

    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size,
                        verbose=verbose, callbacks=callbacks_list)

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # plot model accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(model_name + '_model_acc', dpi=300)
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.savefig(model_name + '_model_loss', dpi=300)
    plt.show()

    class_label = np.concatenate((const_indoor_activity, const_outdoor_activity))

    cnf_matrix = confusion_matrix(y_test, y_pred)

    print("Best accuracy (on validation dataset): %.2f%%" % (accuracy_score(y_test, y_pred) * 100))
    print(cnf_matrix)

    plt.figure()
    fig, ax = plot_confusion_matrix(conf_mat=cnf_matrix,
                                    show_absolute=True,
                                    show_normed=True,
                                    colorbar=True,
                                    class_names=class_label,
                                    figsize=(10, 10))
    plt.savefig(model_name + '_confusion_matrix', dpi=300)
    plt.show()

    # display report
    report_display = classification_report(y_test, y_pred, target_names=class_label, digits=4)
    print("Classification Report:")
    print(report_display)

    # create report and store in csv
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(model_name + '_classification_report')


# %%
# Permutation starts here:
axis_list = ['accX', 'accY', 'accZ']
pos_list = ['LF', 'RF', 'Waist', 'Wrist']

split_method_list = ['TrainTestSplitWithinSubject'
                     # , 'Random', 'DifferentSubjectsInTrainTest'
                     ]
window_list = [512
               # , 256
               ]
wavelet_args_list = [
    {
        "type": "Y",
        "threshold": 2,
        "wavedec_options": {"wavelet": "db4", "level": 2},
        "waverec_options": {"wavelet": "db4"}
    }
    # {"type": "N"}

]
window_args_list = [{"type": "by_peaks", "find_peaks_options": {"prominence": 30, "height": 20}}]

feature_list_list = [
    ['accX_LF', 'accY_LF', 'accZ_LF', 'accX_RF', 'accY_RF', 'accZ_RF',
     'accX_Waist', 'accY_Waist', 'accZ_Waist', 'accX_Wrist', 'accY_Wrist', 'accZ_Wrist']
]

count = 0
for split_method in split_method_list:
    for window in window_list:
        for wavelet_args in wavelet_args_list:
            for window_args in window_args_list:
                for feature_list in feature_list_list:
                    for feature in feature_list:
                        prepare_single_feature_to_csv(feature, split_method, window,
                                                      wavelet_args, window_args)

                    X_train, y_train, X_test, y_test = load_dataset(feature_list, window, wavelet_args, window_args,
                                                                    split_method)

                    print(X_train.shape)
                    print(y_train.shape)
                    print(X_test.shape)
                    print(y_test.shape)
                    print(X_train.shape[1])
                    print(X_train.shape[2])
                    print(y_train.shape[1])

                    count = count + 1
                    model_name = 'model_v' + str(count)
                    train_predict_plot(X_train, y_train, X_test, y_test, model_name=model_name, verbose=1, epochs=5,
                                       batch_size=128)
