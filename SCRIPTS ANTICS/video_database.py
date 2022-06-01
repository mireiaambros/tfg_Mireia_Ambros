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
video_list = []
video_list2 = []
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
    video_ok = 0
    for folder in os.listdir(root):
        if os.path.isdir(os.path.join(root, folder)):
            for file in sorted(os.listdir(root + folder)):
                if file.find("mp4")!=-1:
                    video_path = root + folder + "/" + file
                    prediction = process_video_segment(video_path, model)
                    cont = 0
                    ok = True
                    for i in prediction:
                        if folder == "BikeU" or folder == "BikeBi":
                            folder3 = "bike"
                        else:
                            folder3 = folder

                        if folder3 == class_dict[i]:
                            video_list.append((file, cont, folder3, class_dict[i], "Yes"))
                        else:
                            video_list.append((file, cont, folder3, class_dict[i], "No"))
                            ok = False
                        cont += 1
                    if ok == True:
                        for i in prediction:
                            video_list2.append("Yes")
                        video_ok +=1
                    else:
                        for i in prediction:
                            video_list2.append("No")
                    prediction.clear()  
            df = pd.DataFrame(video_list, video_list2, columns=['Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment'])
            df.insert(loc = 5, column='Correct prediction video', value=video_list2)
    df.to_csv('video_database_df.csv', index=None, columns=None)
    print(video_ok) #Con T=2 da 6 videos correctos

if(__name__ == '__main__'):
    main()