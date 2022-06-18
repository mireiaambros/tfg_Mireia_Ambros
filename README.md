TFG - Mireia Ambros: micromobility
===================================================
This project consists of the creation of video databases of light mobility vehicles and the implementation of Deep Learning algorithms that detect the type of road to increase the urban safety of micromobility vehicles.  

The scripts of this project are divided into two blocks:
* creation_database
* evaluation_models

creation_database
-------
Creation of the structured video **database** and the representation of the route of each video in an interactive **map**.

SCRIPTS: `database2021.py`, `query.py`

evaluation_models
-------
**Evaluation** at segment and video level of 2D models: **ShuffleNet V2** and **MobileNet V3**.

SCRIPTS: `splitdatabase.py`, `modelsSegment.py`, `modelsSegment2.py` (optional), `modelsVideo.py` 