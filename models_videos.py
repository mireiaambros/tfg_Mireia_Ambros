"""  
This script

Usage:
  models_videos.py [--modelType=<mt>] [--segmentDuration=<sd>]
  models_videos.py -h | --help
  
Options:
  --modelType=<mt>        Model used (3d or 2d) [default: 3d]
"""
from docopt import docopt
import pandas as pd
from models import video_list

def main():
    video_list2 = []
    cont = []
    first = True
    ok = True
    video_ok = 0
    num_videos = 0
    if tipus_model == '2d':
        models = pd.read_csv("/home/usuaris/imatge/mireia.ambros/models_2d_df.csv")
    if tipus_model == '3d':
        models = pd.read_csv("/home/usuaris/imatge/mireia.ambros/models_3d_df.csv")

    for row in models.index:
        if models.loc[row, 'Num of segment'] == 0 and first == True:
            if models.loc[row, 'Correct prediction'] == "No":
                ok = False
            cont.append(1)
            first = False
        
        elif models.loc[row, 'Num of segment'] != 0 and first == False:
            if models.loc[row, 'Correct prediction'] == "No":
                ok = False
            cont.append(1)
            if row == (models.index.size - 1):
                if ok == True:
                    for i in cont:
                        video_list2.append("Yes")
                    video_ok += 1
                else:
                    for i in cont:
                        video_list2.append("No")
                num_videos += 1

        elif models.loc[row, 'Num of segment'] == 0 and first == False:
            if ok == True:
                for i in cont:
                    video_list2.append("Yes")
                video_ok += 1
            else:
                for i in cont:
                    video_list2.append("No")
            num_videos += 1
            cont.clear()
            ok = True
            if models.loc[row, 'Correct prediction'] == "No":
                ok = False
            cont.append(1)
            first=False
    
    models['Correct prediction video'] = video_list2
    if tipus_model == '2d':
        models.to_csv('models_videos_2d_df.csv', index=None, columns=None)
    if tipus_model == '3d':
        models.to_csv('models_videos_3d_df.csv', index=None, columns=None)
    print("Num correct videos: ", video_ok)
    print("Num of videos: ", num_videos)
    print("Accuracy (video evaluation): ", round((video_ok/num_videos)*100, 2), "%")

if(__name__ == '__main__'):
     # read arguments
    args = docopt(__doc__)
    tipus_model  = args["--modelType"]
    main()