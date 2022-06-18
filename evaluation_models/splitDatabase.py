import os
import shutil
import pandas as pd

root = "/home/usuaris/imatge/morros/work_fast/mobilitat/ridesafe/barcelona/split_videos/downsampled_low/shared/"
dest = "/home/usuaris/imatge/mireia.ambros/val2022/shared/"

videos = pd.read_csv("/home/usuaris/imatge/mireia.ambros/validation.csv", sep=' ', header=None, names=['Filename', 'num'])

for file_name in os.listdir(root):
    full_file_name = os.path.join(root, file_name)
    if os.path.isfile(full_file_name):
        for row in videos.index:
            a = videos.loc[row, 'Filename']
            if a.find(file_name)!=-1:
                shutil.copy(full_file_name, dest)