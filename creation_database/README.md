creation_database
========

**`database.py`**

DESCRIPTION: 

This script creates the table that contains the information (metadata, weather, part of the day and geolocation) of all the videos in the database using Pandas. It also creates a map with the routes of all the videos using Folium.

INPUT CONFIGURATION:
- train: path where it is located the csv file containing the list of train videos
- val: path where it is located the csv file containing the list of validation videos
- test: path where it is located the csv file containing the list of test videos
- root: path where it is located all videos (training, validation, test or other videos). Videos that do not belong to the above train, validation and test lists, even if they are in this path, will not be added to the database.
- meteo: path where it is located the weather information extracted from Meteocat open data
- sun_2021 and sun_2022: path where sunrise and sunset information is located

OUTPUT CONFIGURATION:
- CSV FILE: Line 247
- HTML MAP: Line 286

NOTES: 
- Now the map is created with daytime routes in orange and nighttime routes in black. Configurable from line 208 to 211. 
- If you do not want the map to show the districts of Barcelona highlighted, you must comment the code of the lines 260-284.

USAGE: 
>python database.py

**`query.py`**

DESCRIPTION: 

This script makes a query of the database table with all the videos. This query can be done with all the fields. It creates a new table and a map that shows only the routes that satisfy the query condition.

INPUT CONFIGURATION:
- df_query: The desired query

OUTPUT CONFIGURATION:
- CSV FILE: Line 11
- HTML MAP: Line 24

USAGE:
>python query.py