import os
from glob import glob
import numpy as np

base = '/HDD1/sung/dataset/dongwoon/val_0.5'

def select(label_dict):
    label_list_1 = label_dict['1'].tolist()

    num_0 = len(label_dict['0'])
    num_1 = len(label_dict['1'])

    if int(num_1 / 2) > num_0:
        label_list_0 = label_dict['0'].tolist()
        label_list_2 = label_dict['-1'][np.random.choice(range(len(label_dict['-1'].tolist())), \
                                        num_1 - num_0, replace=False).tolist()].tolist()
    else:
        label_list_0 = label_dict['0'][np.random.choice(range(len(label_dict['0'].tolist())), \
                                       int(num_1 * 0.7), replace=False).tolist()].tolist()
        label_list_2 = label_dict['-1'][np.random.choice(range(len(label_dict['-1'].tolist())), \
                                       int(num_1 * 0.1), replace=False).tolist()].tolist()
    return label_list_0, label_list_2, label_list_1

# For Event
event_dict = dict(np.load(os.path.join(base, 'meta_dict_event.npz')))
np.savez(os.path.join(base, 'meta_dict_event_pre.npz'),**{'0':select(event_dict)[0], '-1':select(event_dict)[1], '1': select(event_dict)[2]})

# For Voice
voice_dict = dict(np.load(os.path.join(base, 'meta_dict_voice.npz')))
np.savez(os.path.join(base, 'meta_dict_voice_pre.npz'),**{'0':select(voice_dict)[0], '-1':select(voice_dict)[1], '1': select(voice_dict)[2]})
