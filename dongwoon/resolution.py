import os
import glob
import numpy as np
from glob import glob
from scipy.io.wavfile import read
from collections import defaultdict
from tqdm import tqdm
import pickle
import random
import argparse

# Labeling
def label_set(load_folder, save_folder, label_len, criterion, remove_other=False):
    os.makedirs(save_folder, exist_ok=True)

    load_list = glob(os.path.join(load_folder, '*.npz'))

    meta_dict = {'0':[], '1':[], '-1':[]}
    for file_name in tqdm(load_list):
        new_dict = {}
        save_name = os.path.basename(file_name)

        one_sample = dict(np.load(file_name))

        # Audio file
        audio_list = np.array(one_sample['audio']).tolist()
        audio_list = np.array_split(audio_list, int(len(one_sample['audio']) // 8000))

        # label file
        label = one_sample['label']
        label_list = []

        if len(label) % label_len == 0:
            label_num = int(len(label) // label_len)
        else:
            label_num = int(len(label) // label_len) + 1

        label = np.array_split(label, label_num)

        for l in label:
            if len(l) != label_len:
                continue
            else:
                out = criterion(l, remove_other)
                label_list.append(np.array([out] * label_len))

        if len(label_list) == 0:
            continue
        else:
            label_list = np.concatenate(label_list, axis=0)

        label_list = label_list.tolist()

        # Matching the label and audio sample num
        audio_list = audio_list[:len(label_list)]

        new_dict['audio'] = audio_list
        new_dict['label'] = label_list

        # Save the meta label
        for ix, l in enumerate(label_list):
            meta_dict[str(l)].append((os.path.basename(file_name), ix))

        if len(label_list) > 0:
            np.savez(os.path.join(save_folder, save_name), **new_dict)
        else:
            continue

    dir, name = '/'.join(save_folder.split('/')[:-1]), save_folder.split('/')[-1]
    np.savez(os.path.join(dir, 'meta_dict_%s.npz' %name), **meta_dict)

def event_criterion(label, remove_other=False):
    label = np.array(label)

    # Remove others label
    if remove_other:
        index = np.where(label == 3)[0]
        label[index] = 9999
        remain_index = list(set(range(len(label))) - set(index))
    else:
        remain_index = list(range(len(label)))

    # Select the optimal labels from the remainders
    remain = label[remain_index]

    if (1 in remain) or (2 in remain):
        ind = 1
    elif (3 in remain):
        ind = -1
    else:
        ind = 0
    return ind

def voice_criterion(label, remove_other=False):
    label = np.array(label)

    if remove_other:
        # Remove others label
        index = np.where(label == 3)[0]
        label[index] = 9999

        remain_index = list(set(range(len(label))) - set(index))
    else:
        remain_index = list(range(len(label)))

    # Select the optimal labels from the remainders
    remain = label[remain_index]

    if (0 in remain) or (2 in remain):
        ind = 1
    elif (3 in remain):
        ind = -1
    else:
        ind = 0
    return ind

def no_criterion(label):
    label = np.array(label)

    # Remove others label
    len_0 = len(np.where(label == 0)[0])
    len_1 = len(np.where(label == 1)[0])

    cri = len_1 - len_0
    if cri == 0:
        ind = 9999
    elif cri > 0:
        ind = 1
    else:
        ind = 0
    return ind


if __name__=='__main__':
    # option
    arg = argparse.ArgumentParser()
    arg.add_argument('--base', type=str, default='/HDD1/sung/dataset/dongwoon')
    args = arg.parse_args()

    # initialize
    base = args.base

    # 0.5 초 resolution
    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_0.5', 'event'),
              label_len=1, criterion=event_criterion, remove_other=False)

    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_0.5', 'voice'),
              label_len=1, criterion=voice_criterion, remove_other=False)

    label_set(os.path.join(base,'val'),
              os.path.join(base, 'val_0.5', 'event'),
              label_len=1, criterion=event_criterion, remove_other=False)

    label_set(os.path.join(base,'val'),
              os.path.join(base, 'val_0.5', 'voice'),
              label_len=1, criterion=voice_criterion, remove_other=False)

    # 2 초 resolution
    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_2.0', 'event'),
              label_len=4, criterion=event_criterion, remove_other=False)

    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_2.0', 'voice'),
              label_len=4, criterion=voice_criterion, remove_other=False)

    # 4 초 resolution
    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_4.0', 'event'),
              label_len=8, criterion=event_criterion, remove_other=False)

    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_4.0', 'voice'),
              label_len=8, criterion=voice_criterion, remove_other=False)

    # 8 초 resolution
    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_8.0', 'event'),
              label_len=16, criterion=event_criterion, remove_other=False)

    label_set(os.path.join(base,'train'),
              os.path.join(base, 'label_8.0', 'voice'),
              label_len=16, criterion=voice_criterion, remove_other=False)
