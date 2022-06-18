TFG - Mireia Ambros: micromobility
===================================================
This project consists of the creation of video databases of light mobility vehicles and the implementation of Deep Learning algorithms that detect the type of road to increase the urban safety of micromobility vehicles.  

The scripts of this project are divided into two blocks:
creation_database
-------
* Creation of the structured video **database** and the representation of the route of each video in an interactive **map**.

SCRIPTS: `database2021.py`, `query.py`

evaluation_models
-------
* **Evaluation** at segment and video level of 2D models: **ShuffleNet V2** and **MobileNet V3**.
SCRIPTS: `splitdatabase.py` (optional), `modelsSegment.py`, `modelsSegment2.py` (optional), `modelsVideo.py` 

SCRIPTS
-------

`database.py`

Usage: 
>python database.py

`query.py`

Usage:
>python query.py

`modelsSegment.py`

Usage:
>srun --gres=gpu:1 --time=15:15:00 -c 10 --mem 20G python modelsSegment.py --modelType='2d' --segmentDuration=2 --framesPerSecond=3
>
The parameters segmentDuration (value of T seconds of each segment) and framesPerSecond (value of F frames per second that you use for the prediction in each segment) can be adjusted by the user when running the script.

`modelsVideo.py`

Usage:
>python modelsVideo.py --modelType='2d'

`splitDatabase.py`

Usage:
>python splitDatabase.py