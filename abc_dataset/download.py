import subprocess
import os
from glob import glob
from tqdm import tqdm

def download():
    subprocess.call('cat file_path.txt | xargs -n 2 -P 8 sh -c \'wget --no-check-certificate $0 -O /data/sung/dataset/abc_dataset/$1\'', shell=True)

def strip_title(base_dir, target_dir, type='step'):
    file_list = glob(os.path.join(base_dir, '*_%s_*' %type))

    save_dir = os.path.join(target_dir, type)
    os.makedirs(save_dir, exist_ok=True)

    for old_name in tqdm(file_list):
        new_name = os.path.join(save_dir, os.path.basename(old_name).strip())

        subprocess.call('cp %s %s' % (old_name, new_name), shell=True)

def unzip(target_dir, type='step'):
    save_dir = os.path.join(target_dir, type)
    file_list = glob(os.path.join(save_dir, '*.7z'))
    print(file_list)
    exit()
    for file in tqdm(file_list):
        subprocess.call('7z x %s -o%s' %(file, save_dir), shell=True)
        subprocess.call('rm -rf %s' %file, shell=True)


if __name__=='__main__':
    # 1. Download the ABC-dataset
    print('Download .7z files')
    # download()

    # 2. Strip the title and split the step / meta / ... into each types
    print('Strip the title')
    # base_dir = '/data/sung/dataset/abc_dataset'
    # target_dir = '/data/sung/dataset/abc_cad'
    # type='meta'

    # strip_title(base_dir, target_dir, type)

    # 3. Unzip Files
    print('Unzip .7z files')
    # target_dir = '/data/sung/dataset/abc_cad'
    target_dir = '/data/Seongju/dset/ABC'
    type = 'step'

    unzip(target_dir, type)