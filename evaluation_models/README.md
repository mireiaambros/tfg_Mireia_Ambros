evaluation_models
--------

`splitDatabase.py`

Usage:
>python splitDatabase.py

`modelsSegment.py`

Usage:
>srun --gres=gpu:1 --time=15:15:00 -c 10 --mem 20G python modelsSegment.py --modelType='2d' --segmentDuration=2 --framesPerSecond=3
>
The parameters segmentDuration (value of T seconds of each segment) and framesPerSecond (value of F frames per second that you use for the prediction in each segment) can be adjusted by the user when running the script.

`modelsSegment2.py`

Usage:
>python modelsVideo.py --modelType='2d'

`modelsVideo.py`

Usage:
>python modelsVideo.py --modelType='2d'