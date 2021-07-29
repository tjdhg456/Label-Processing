import os
import numpy as np
from glob import glob

base = '/HDD1/sung/dataset/dongwoon'

all_file_list = glob(os.path.join(base, 'original', '*.npz'))
file_list = np.unique([os.path.basename(f).split('_')[1] for f in all_file_list])

# train / val split
train_list = np.random.choice(file_list.tolist(), int(len(file_list) * 0.75), replace=False).tolist()
val_list = list(set(file_list) - set(train_list))

# Move Files
for train in train_list:
    tr_ori_name = os.path.join(base, 'original', 'pubg_%s_*.npz' %train)

    tr_new_folder = os.path.join(base, 'train')
    os.makedirs(tr_new_folder, exist_ok=True)
    os.system('cp -r %s %s' %(tr_ori_name, tr_new_folder))

for val in val_list:
    val_ori_name = os.path.join(base, 'original', 'pubg_%s_*.npz' %val)

    val_new_folder = os.path.join(base, 'val')
    os.makedirs(val_new_folder, exist_ok=True)
    os.system('cp -r %s %s' %(val_ori_name, val_new_folder))

