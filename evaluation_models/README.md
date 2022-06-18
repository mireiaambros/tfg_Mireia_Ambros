evaluation_models
========

**`splitDatabase.py`**

DESCRIPTION:

This script allows the database to be separated into three different sets (training, validation and test set) in order to test the models. Copies the videos belonging to a set to another folder. 

INPUT CONFIGURATION:
- root: path where it is located all videos (training, validation, test or other videos)
- dest: path where you want to copy the videos
- video: path where the csv file containing the list of videos you want to copy is located (training, validation or test set)

USAGE:
>python splitDatabase.py

**`modelsSegment.py`**

DESCRIPTION:

This script separates the videos into segments of T seconds and makes a prediction for each segment using only F number of frames per second. The predictions are saved in a CSV file with the following information: 'Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment' (it is checked if the segment has been predicted correctly). A confusion matrix is also created.

INPUT CONFIGURATION:
- model: path where it is located the model you want to evaluate. There are three possible models in this script: 2d (ShuffleNet v2 trained with 2021 database), 2d_2022 (MobiliNet v3 trained with 2022 database) and 3d (X3D trained with 2021 database)
- root: path where the videos of the set to be evaluated are located (folder obtained after executing *splitDatabase.py*)

OUTPUT CONFIGURATION:
- CSV FILE with the information 'Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment': Lines 286-291
- PNG: Confusion matrix. Lines 481-486

USAGE:
>srun --gres=gpu:1 --time=15:15:00 -c 10 --mem 20G python modelsSegment.py --modelType='2d' --segmentDuration=2 --framesPerSecond=3
>
The parameters segmentDuration (value of T seconds of each segment) and framesPerSecond (value of F frames per second that you use for the prediction in each segment) can be adjusted by the user when running the script. Also choose the model that will be used to make the prediction (2d, 2d_2022, 3d)

**`modelsSegment2.py`** (optional)

DESCRIPTION:

This script is only used to check that segment level predictions made by other teams give the same results. You get the same output as *modelsSegment.py* allowing you to evaluate the model at the video level with the *modelsVideo.py* script.

INPUT CONFIGURATION:
- root: path where the results of the segment predictions are located

OUTPUT CONFIGURATION:
- CSV FILE with the information 'Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment': Line 27

USAGE:
>python modelsSegment2.py

**`modelsVideo.py`**

DESCRIPTION:

This script performs a video-level prediction. It discards any video that has at least one incorrectly predicted segment. It returns the same table as modelsSegment.py but with an additional column that says whether the video is detected as correct or incorrect. It also prints the accuracy of the database at video level and for each of the classes.

INPUT CONFIGURATION:
- models: path where you can find the results with the predictions of each segment of the chosen models. Remember that there are three possible models in this script: 2d (ShuffleNet v2 trained with 2021 database), 2d_2022 (MobiliNet v3 trained with 2022 database) and 3d (X3D trained with 2021 database)

OUTPUT CONFIGURATION:
- CSV FILE with the information 'Filename', 'Num of segment', 'Actual', 'Prediction', 'Correct prediction segment', 'Correct prediction video': Line 84, 104 or 121 (depending on the model used to make the prediction. Note that in order to save the results and print the accuracy for each class it should be taken into account that in 2022 there is no more crosswalk class) 

USAGE:
>python modelsVideo.py --modelType='2d'