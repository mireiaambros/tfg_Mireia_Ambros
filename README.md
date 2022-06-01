TFG - Mireia Ambros: micromobility
===================================================
Los scripts de este proyecto se dividen en dos bloques:
* Creación de la **base de datos de vídeos** estructurada y la representación de la ruta de cada vídeo en un **mapa** interactivo
SCRIPTS: `database2021.py`, `database2022.py`, `query.py`
* **Evaluación** a nivel de segmento y de vídeo de los modelos 2D: **ShuffleNet V2** y **MobileNet V3**
SCRIPTS: `modelsSegment.py`, `modelsVideo.py`, `splitdatabase.py`
------------------
`database2021.py` y `database2022.py`
Para ejecutar el programa: 
>python database2021.py
>python database2022.py
------------------
`query.py`
Para ejecutar el programa: 
>python query.py
------------------
`modelsSegment.py`
Para ejecutar el programa: 
>srun --gres=gpu:1 --time=15:15:00 -c 10 --mem 20G python models.py --modelType='2d' --segmentDuration=2 --framesPerSecond=3
>
Los parámetros segmentDuration (valor de T segundos de cada segmento) y framesPerSecond (valor de F frames por segundo que coges para hacer la predicción en cada segmento) los puede ajustar el usuario a la hora de correr el script.
------------------
`modelsVideo.py`
Para ejecutar el programa: 
>python modelsVideo.py --modelType='2d'
------------------
`splitDatabase.py`
Para ejecutar el programa: 
>python splitDatabase.py