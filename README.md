TFG - Mireia Ambros: micromobility
===================================================
The scripts of this project are divided into two blocks:
* Creation of the structured video **database** and the representation of the route of each video in an interactive **map**.
SCRIPTS: `database2021.py`, `query.py`
* **Evaluation** at segment and video level of 2D models: **ShuffleNet V2** and **MobileNet V3**.
SCRIPTS: `splitdatabase.py` (optional), `modelsSegment.py`, `modelsVideo.py` 

SCRIPTS
-------

`database.py`

Para ejecutar el programa: 
>python database.py

`query.py`

Para ejecutar el programa: 
>python query.py

`modelsSegment.py`

Para ejecutar el programa: 
>srun --gres=gpu:1 --time=15:15:00 -c 10 --mem 20G python modelsSegment.py --modelType='2d' --segmentDuration=2 --framesPerSecond=3
>
Los parámetros segmentDuration (valor de T segundos de cada segmento) y framesPerSecond (valor de F frames por segundo que coges para hacer la predicción en cada segmento) los puede ajustar el usuario a la hora de correr el script.

`modelsVideo.py`

Para ejecutar el programa: 
>python modelsVideo.py --modelType='2d'

`splitDatabase.py`

Para ejecutar el programa: 
>python splitDatabase.py