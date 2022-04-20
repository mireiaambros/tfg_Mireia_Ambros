import torch
import torchvision
from types import SimpleNamespace

import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

import os
import pandas as pd
import math

root = "/home/usuaris/imatge/mireia.ambros/videos/"
folder2 = []
y_pred = []
video_list = []
pred_class = []
T = 2 #number of seconds of each T segment

def process_video_segment(path, model):
   
    #define transformation parameters
    transform = transforms.Compose([
        #transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    cap = cv2.VideoCapture(path) #Creating a video capture object

    # Video statistics
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) #Extreu info del numero de frames
    if fps != 0:
      video_length = frame_length/fps
    decimla, frames_T = math.modf(fps * T)
    cont_frames_T = 0
    
    # Variables
    prob_list = []
    current_frame = 0

    # Processment of the video
    try :
      while True:
        # Reads a frame of a video
        ret, frame = cap.read() 

        if ret and current_frame < frame_length:

            # Apply a transform to normalize the image from video input
            image_data = transform(Image.fromarray(frame))
                      
            # Pass the input clip through the model
            inputs = image_data
            inputs = inputs.to("cuda")

            # Prediction 
            preds = model(inputs[None, ...]) 

            # Saves predictions for each image
            post_act = torch.nn.Softmax(dim=1) 
            preds = post_act(preds.cpu()).detach().numpy()

            if cont_frames_T < frames_T:
                prob_list.append(preds)
                cont_frames_T += 1
            else:
                preds = sum(prob_list) #Sums the probabilities of each image in the segments of T seconds
                pred_class.append(np.argmax(preds)) #Get the predicted classes of the segment
                cont_frames_T = 0
                prob_list.clear()
            
            current_frame += 1
                      
        else:
            #No more frames
            preds = sum(prob_list) #Sums the probabilities of each image in the segments of T seconds
            pred_class.append(np.argmax(preds)) #Get the predicted classes of the segment
            cap.release()
            break

    except:
        print("Video has ended..") #if any error occurs then this block of code will run

    return pred_class


# dictionary to identify the classes
class_dict = {
    0: "sidewalk",
    1: "road",
    2: "shared",
    3: "bike",
    4: "crosswalk"
}

def main():
    # some arguments needed
    hardcoded_args = SimpleNamespace(**{
        "video_means": [0.45, 0.45, 0.45],
        "video_stds": [0.225, 0.225, 0.225],
        "transform_params": {
            "side_size": 160,
            "crop_size": 160,
            "num_frames": 4,
            "sampling_rate": 12
        },
    })
    # load the model
    model = torch.jit.load("/home/usuaris/imatge/mireia.ambros/shufflenet1_multi.pt").to("cuda")

    # load some video
    for folder in os.listdir(root):
        if os.path.isdir(os.path.join(root, folder)):
            for file in sorted(os.listdir(root + folder)):
                if file.find("mp4")!=-1:
                    video_path = root + folder + "/" + file
                    prediction = process_video_segment(video_path, model)
                    cont = 0
                    for i in prediction:
                        if folder == "BikeU" or folder == "BikeBi":
                            folder3 = "bike"
                            folder2.append(folder3)
                        else:
                            folder3 = folder
                            folder2.append(folder3)

                        if folder3 == class_dict[i]:
                            video_list.append((file, cont, folder3, class_dict[i], "Yes"))
                        else:
                            video_list.append((file, cont, folder3, class_dict[i], "No"))
                        cont += 1
                        y_pred.append(class_dict[i])
                    prediction.clear()         
            df = pd.DataFrame(video_list, columns=['Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction'])
    df.to_csv('segment_database_df.csv', index=None, columns=None)

#Matriu de confusio
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

class_list = ['bike', 'crosswalk', 'road', 'shared']

def get_new_fig(fn, figsize=[10,10]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=15,
      lw=0.5, cbar=False, figsize=[10,10], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 12)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
    plt.show()
    plt.savefig('segment_database_plot.png')

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=15, lw=0.5, cbar=False, figsize=[12,12], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[10,10];
    show_null_values = 2
    df_cm = pd.DataFrame(confm, index=class_list, columns=class_list)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)

if(__name__ == '__main__'):
    main()
    plot_confusion_matrix_from_data(folder2, y_pred)