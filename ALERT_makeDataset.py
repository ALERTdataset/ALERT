import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


def load_csv(path):
    df = pd.read_csv(path, header=None)
    mat = df.values
    return mat[1:, 1:]


def make_onehot(label_idx, num_classes):
    onehot = [0] * num_classes
    onehot[label_idx] = 1
    return onehot


def find_label_idx(path):
    for key in label_dict:
        if key in path:
            return label_dict[key]
    return None


def calc_CA(time_mat):
    max_val = 0
    ca = 0
    for row in range(time_mat.shape[0]):
        if time_mat[row][0] > max_val:
            max_val = time_mat[row][0]
            ca = row
    return ca


class ExtendDataset(Dataset):

    def __init__(self, time_file, freq_file, sample_size):

        for time_path, freq_path in zip(time_file, freq_file):

            subject = os.path.basename(time_path).split("_")[0]

            time_mat = load_csv(time_path)
            freq_mat = load_csv(freq_path)

            stride = sample_size // 2
            total_len = time_mat.shape[1]
            n_samples = (total_len - sample_size) // stride + 1

            for task in subject_list:

                if task != subject:
                    continue

                label_idx = find_label_idx(time_path)
                tmp = make_onehot(label_idx, num_classes)

                for i in range(n_samples):

                    start = i * stride
                    end = start + sample_size

                    subject_labels[task][label_idx].append(tmp)

                    if CA_dict[task] == 0:
                        CA_dict[task] = calc_CA(time_mat)

                    apply_crop(task, label_idx, time_mat, freq_mat, start, end)


class MyDataset(Dataset):

    def __init__(self, time_file, freq_file, sample_size):

        loc_count = 0

        for task in subject_list:

            key_task = 'ALERT_train/' + task + '_relax_1_t.csv'
            loc = time_file.index(key_task)

            time_file[loc_count], time_file[loc] = time_file[loc], time_file[loc_count]
            freq_file[loc_count], freq_file[loc] = freq_file[loc], freq_file[loc_count]

            loc_count += 1

        for time_path, freq_path in zip(time_file, freq_file):

            subject = os.path.basename(time_path).split("_")[0]

            time_mat = load_csv(time_path)
            freq_mat = load_csv(freq_path)

            n_samples = time_mat.shape[1] // sample_size

            for task in subject_list:

                if task != subject:
                    continue

                label_idx = find_label_idx(time_path)
                tmp = make_onehot(label_idx, num_classes)

                for i in range(n_samples):

                    start = i * sample_size
                    end = (i + 1) * sample_size

                    subject_labels[task][label_idx].append(tmp)

                    if CA_dict[task] == 0:
                        CA_dict[task] = calc_CA(time_mat)

                    apply_crop(task, label_idx, time_mat, freq_mat, start, end)


def apply_crop(task, label_idx, time_mat, freq_mat, start, end):

    if sys.argv[2] == 'CA':

        subject_time[task][label_idx].append(
            time_mat[(CA_dict[task]-10):(CA_dict[task]+5), start:end]
        )

        subject_freq[task][label_idx].append(
            freq_mat[0:89, start:end]
        )

    elif sys.argv[2] == 'CA_fft':

        time_crop = time_mat[(CA_dict[task]-10):(CA_dict[task]+5), start:end]

        subject_time[task][label_idx].append(time_crop)

        fft = np.fft.fft(time_crop, axis=1)

        subject_freq[task][label_idx].append(abs(fft))

    elif sys.argv[2] == 'cropX':

        subject_time[task][label_idx].append(
            time_mat[rangebin_start:rangebin_end, start:end]
        )

        subject_freq[task][label_idx].append(
            freq_mat[45:89, start:end]
        )

    elif sys.argv[2] == 'cropO':

        subject_time[task][label_idx].append(
            time_mat[rangebin_start:rangebin_end, start:end]
        )

        subject_freq[task][label_idx].append(
            freq_mat[0:89, start:end]
        )

    elif sys.argv[2] == 'RD':

        fft_result = np.fft.fft(
            time_mat[rangebin_start:rangebin_end, start:end],
            axis=1
        )

        subject_time[task][label_idx].append(np.abs(fft_result))

        subject_freq[task][label_idx].append(
            freq_mat[0:89, start:end]
        )


def usage_exam():

    print("===============================================================================================================")
    print("                                 USAGE EXAMPLE")
    print("===============================================================================================================")
    print("python3 ALERT_makeDataset.py common/extend cropO/cropX/CA/RD sample_size")
    print("===============================================================================================================")
    sys.exit()


def make_pickle(crop, sample_size, num_classes):

    for subject in subject_list:

        with open(f'./pickles/{crop}_{subject}_{sample_size}_{num_classes}_time_data.pickle','wb') as f:
            pickle.dump(subject_time[subject], f)

        with open(f'./pickles/{crop}_{subject}_{sample_size}_{num_classes}_freq_data.pickle','wb') as f:
            pickle.dump(subject_freq[subject], f)

        with open(f'./pickles/{crop}_{subject}_{sample_size}_{num_classes}_labels_data.pickle','wb') as f:
            pickle.dump(subject_labels[subject], f)


if __name__ == "__main__":

    label_dict = {'relax':0,'drive':1,'nod':2,'drink':3,'phone':4,'smoke':5,'panel':6}

    num_classes = len(label_dict)

    CA_dict = {s:0 for s in ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11']}

    subject_list = list(CA_dict.keys())

    subject_time = {s:[[] for _ in range(num_classes)] for s in subject_list}
    subject_freq = {s:[[] for _ in range(num_classes)] for s in subject_list}
    subject_labels = {s:[[] for _ in range(num_classes)] for s in subject_list}

    sample_size = int(sys.argv[3])

    try:

        if sys.argv[2] in ['cropO','RD']:
            rangebin_start = 120
            rangebin_end = 171

        elif sys.argv[2] == 'cropX':
            rangebin_start = 0
            rangebin_end = 177

        elif sys.argv[2] == 'CA':
            rangebin_start = 140
            rangebin_end = 171

        elif sys.argv[2] == 'CA_fft':
            rangebin_start = 140
            rangebin_end = 172

        else:
            usage_exam()

    except:
        usage_exam()

    time_files = []
    freq_files = []

    for name in glob.glob("ALERT_train/*_t.csv"):
        time_files.append(name)
        freq_files.append(name[:-6]+"_f.csv")

    if sys.argv[1] == 'common':
        MyDataset(time_files, freq_files, sample_size)

    elif sys.argv[1] == 'extend':
        ExtendDataset(time_files, freq_files, sample_size)

    make_pickle(sys.argv[2], sample_size, num_classes)