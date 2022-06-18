from docopt import docopt
import pandas as pd
import os

root = "/home/usuaris/imatge/mireia.ambros/results2021_val_seg_new/"
num_segment = []
data = []
actual = []
prediction = []


for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in sorted(os.listdir(root + folder)):
            results = pd.read_csv(root + folder + "/" + file, sep=',')
            for row in results.index:
                num_segment = results.loc[row, 'segment']
                actual = results.loc[row, 'gt_class']
                prediction = results.loc[row, 'pred_class']
                if actual == prediction:
                    data.append((file, num_segment, actual, prediction, "Yes"))
                else:
                    data.append((file, num_segment, actual, prediction, "No"))
            
        #df = pd.DataFrame(data, columns=['Num of segment'])
        df = pd.DataFrame(data, columns=['Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment'])
df.to_csv('segment_valShuffleNet2022.csv', index=None, columns=None)