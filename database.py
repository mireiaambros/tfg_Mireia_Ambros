import os
import ffmpeg
import pandas as pd
import datetime
import pytz
import geopandas as gpd   # Geodataset
import osmnx as ox        # City graphs (streets and nodes datasets) https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.save_load.save_graphml
import folium             # Visualization library  
from shapely.geometry import MultiPoint, Point, Polygon, shape   # Handle geometry
import json
import numpy as np
from haversine import haversine, Unit

train = pd.read_csv("/home/usuaris/imatge/mireia.ambros/train2022.csv", sep=' ', header=None, names=['Filename', 'num'])
val = pd.read_csv("/home/usuaris/imatge/mireia.ambros/validation2022.csv", sep=' ', header=None, names=['Filename', 'num'])
test = pd.read_csv("/home/usuaris/imatge/mireia.ambros/test2022.csv", sep=' ', header=None, names=['Filename', 'num'])

root = "/home/usuaris/imatge/morros/work_fast/mobilitat/ridesafe/barcelona/split_videos/ori/"
data = []
midd_lat1 = []
midd_lon1 = []
midd_lat2 = []
midd_lon2 = []

for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in sorted(os.listdir(root + folder)):
          for row in train.index:
            a = train.loc[row, 'Filename']
            if a.find(file)!=-1 and file.find("mp4")!=-1 and file!=".VID_20220502_155948_2.mp4.gkqOL0":
              info = ffmpeg.probe(root + folder + "/" + file)
              video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video') #Extract metadata using ffmpeg
              time = (video_info['duration'])
              num_frame = int(video_info['nb_frames'])
              date = (video_info['tags']['creation_time'])
              date = date.replace('T', ' ')
              date = date.replace('.000000Z', '') #Date format YYYY-MM-DD (HH:MM:SS), Zulu time at the zero meridian
              print(date)
              date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #Convert the date to a datetime object
              timezone = pytz.utc
              date_utc = timezone.localize(date_obj)
              date_local = date_utc.astimezone(pytz.timezone("Europe/Madrid")) #Shows the local time and the correction made to the Zulu time zone
              print(date_local)
              data.append((folder, file, root + folder, date_local, round(float(time), 2), round(num_frame/float(time), 2), "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information"))
          for row in val.index:
            b = val.loc[row, 'Filename']
            if b.find(file)!=-1 and file.find("mp4")!=-1 and file!=".VID_20220502_155948_2.mp4.gkqOL0":
              info = ffmpeg.probe(root + folder + "/" + file)
              video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video') #Extract metadata using ffmpeg
              time = (video_info['duration'])
              num_frame = int(video_info['nb_frames'])
              date = (video_info['tags']['creation_time'])
              date = date.replace('T', ' ')
              date = date.replace('.000000Z', '') #Date format YYYY-MM-DD (HH:MM:SS), Zulu time at the zero meridian
              print(date)
              date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #Convert the date to a datetime object
              timezone = pytz.utc
              date_utc = timezone.localize(date_obj)
              date_local = date_utc.astimezone(pytz.timezone("Europe/Madrid")) #Shows the local time and the correction made to the Zulu time zone
              print(date_local)
              data.append((folder, file, root + folder, date_local, round(float(time), 2), round(num_frame/float(time), 2), "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information")) 
          for row in test.index:
            c = test.loc[row, 'Filename']
            if c.find(file)!=-1 and file.find("mp4")!=-1 and file!=".VID_20220502_155948_2.mp4.gkqOL0":
              info = ffmpeg.probe(root + folder + "/" + file)
              video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video') #Extract metadata using ffmpeg
              time = (video_info['duration'])
              num_frame = int(video_info['nb_frames'])
              date = (video_info['tags']['creation_time'])
              date = date.replace('T', ' ')
              date = date.replace('.000000Z', '') #Date format YYYY-MM-DD (HH:MM:SS), Zulu time at the zero meridian
              print(date)
              date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #Convert the date to a datetime object
              timezone = pytz.utc
              date_utc = timezone.localize(date_obj)
              date_local = date_utc.astimezone(pytz.timezone("Europe/Madrid")) #Shows the local time and the correction made to the Zulu time zone
              print(date_local)
              data.append((folder, file, root + folder, date_local, round(float(time), 2), round(num_frame/float(time), 2), "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information"))    
        df = pd.DataFrame(data, columns=['Road type', 'Filename', 'Path', 'Date', 'Duration [s]', 'Resolution [fps]', 'First Latitude', 'First Longitude', 'Last Latitude', 'Last Longitude', 'District', 'Distance [m]', 'Speed [km/h]', 'Part of the day', 'Temperature [C]', 'Rain', 'Wind'])
        
for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in sorted(os.listdir(root + folder)):
          if file.find("idx")!=-1 and os.stat(root + folder + "/" + file).st_size!=0 and file!="VID_20220503_230504_1.idx" and file!="VID_20220428_164114_1.idx":
            filename = file.replace('.idx', '.')
            if len(df.index[df['Filename'] == filename + "mp4"])!=0:
              this_row = df.index[df['Filename'] == filename + "mp4"].tolist() #In which row (file) the location should be inserted
              # Get first and last coordinates of idx file
              with open(root + folder + "/" + file, "r") as f:
                line = f.readlines()
                first_lat = line[0].split(' ')[4]
                first_lon = line[0].split(' ')[5]
                last_lat = line[-1].split(' ')[4]
                last_lon = line[-1].split(' ')[5]
                midd_lat1.append(float(line[-100].split(' ')[4]))
                midd_lon1.append(float(line[-100].split(' ')[5]))
                midd_lat2.append(float(line[-50].split(' ')[4]))
                midd_lon2.append(float(line[-50].split(' ')[5]))
              df.iloc[this_row, 6] = round(float(first_lat), 6)
              df.iloc[this_row, 7] = round(float(first_lon), 6) 
              df.iloc[this_row, 8] = round(float(last_lat), 6) 
              df.iloc[this_row, 9] = round(float(last_lon), 6)

#Weather and part of the day information
meteo = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Dades_meteorologiques_de_la_XEMA.csv", sep = ',')

sun_2010 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2020.csv")
sun_2021 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2021.csv")
sun_2022 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2022.csv")

def rain(value):
  if value == 0:
    return "No rain"
  if value > 0 and value <= 2:
    return "Weak"
  if value > 2 and value <= 15:
    return "Moderate"
  if value > 15 and value <= 30:
    return "Strong"
  if value > 30 and value <= 60:
    return "Very heavy"
  if value > 60:
    return "Torrential"

def wind(value):
  if (value*3.6) < 15:
    return "No wind"
  if (value*3.6) >= 15 and (value*3.6) <= 40:
    return "Moderate"
  if (value*3.6) > 40 and (value*3.6) <= 70:
    return "Strong"
  if (value*3.6) > 70 and (value*3.6) <= 120:
    return "Very strong"
  if (value*3.6) > 120:
    return "Hurricanes"

for row in df.index:
  if df.loc[row, 'Date'].year == 2020:
    info_year = sun_2010
  if df.loc[row, 'Date'].year == 2021:
    info_year = sun_2021
  if df.loc[row, 'Date'].year == 2022:
    info_year = sun_2022
  for n in info_year.index:
    date_obj3 = datetime.datetime.strptime(info_year.loc[n, 'Fecha'], '%d/%m/%Y')
    if date_obj3.day == df.loc[row, 'Date'].day and date_obj3.month == df.loc[row, 'Date'].month and date_obj3.year == df.loc[row, 'Date'].year:
      date_obj4 = datetime.datetime.strptime(info_year.loc[n, 'Salida de sol'], '%I:%M:%S %p') + datetime.timedelta(0,-1800)
      date_obj5 = datetime.datetime.strptime(info_year.loc[n, 'Puesta de sol'], '%I:%M:%S %p') + datetime.timedelta(0, 1800)
      date_obj6 = date_obj4 + datetime.timedelta(0,3600)
      date_obj7 = datetime.datetime.strptime("11:50:59", '%H:%M:%S')
      date_obj8 = datetime.datetime.strptime("12:10:59", '%H:%M:%S')
      date_obj9 = datetime.datetime.strptime("17:00:59", '%H:%M:%S')
      if df.loc[row, 'Date'].time() >= date_obj4.time() and df.loc[row, 'Date'].time() <= date_obj6.time():
          df.loc[row, 'Part of the day'] = "Early morning"
      if df.loc[row, 'Date'].time() > date_obj6.time() and df.loc[row, 'Date'].time() <= date_obj7.time():
          df.loc[row, 'Part of the day'] = "Morning"
      if df.loc[row, 'Date'].time() > date_obj7.time() and df.loc[row, 'Date'].time() <= date_obj8.time():
          df.loc[row, 'Part of the day'] = "Noon"
      if df.loc[row, 'Date'].time() > date_obj8.time() and df.loc[row, 'Date'].time() <= date_obj9.time():
          df.loc[row, 'Part of the day'] = "Afternoon"
      if df.loc[row, 'Date'].time() > date_obj9.time() and df.loc[row, 'Date'].time() <= date_obj5.time():
          df.loc[row, 'Part of the day'] = "Evening"
      if df.loc[row, 'Date'].time() < date_obj4.time() or df.loc[row, 'Date'].time() > date_obj5.time():
        df.loc[row, 'Part of the day'] = "Night"
#  for i in meteo.index:
#    date_found = False
#    date_obj2 = datetime.datetime.strptime(meteo.loc[i, 'DATA_LECTURA'], '%d/%m/%Y %I:%M:%S %p')
#    if date_obj2.day == df.loc[row, 'Date'].day and date_obj2.month == df.loc[row, 'Date'].month and date_obj2.year == df.loc[row, 'Date'].year and date_found == False:
#      date_found =  True
#      if date_obj2.hour == df.loc[row, 'Date'].hour:
#        if meteo.loc[i, 'DATA_LECTURA'].find("AM")!=-1:
#          info_hour = "AM"
#        if meteo.loc[i, 'DATA_LECTURA'].find("PM")!=-1:
#          info_hour = "PM"
#        if meteo.loc[i, 'DATA_LECTURA'].find(info_hour)!=-1:
#          if df.loc[row, 'Date'].minute < 15 and date_obj2.minute == 0:
#            if meteo.loc[i, 'CODI_VARIABLE'] == 32:
#              df.loc[row, 'Temperature [C]'] = meteo.loc[i, 'VALOR_LECTURA']
#            if meteo.loc[i, 'CODI_VARIABLE'] == 35:
#              df.loc[row, 'Rain'] = rain(meteo.loc[i, 'VALOR_LECTURA'])
#            if meteo.loc[i, 'CODI_VARIABLE'] == 30:
#              df.loc[row, 'Wind'] = wind(meteo.loc[i, 'VALOR_LECTURA'])
#          if df.loc[row, 'Date'].minute >=15 and df.loc[row, 'Date'].minute < 45 and date_obj2.minute == 30:
#            if meteo.loc[i, 'CODI_VARIABLE'] == 32:
#              df.loc[row, 'Temperature [C]'] = meteo.loc[i, 'VALOR_LECTURA']
#            if meteo.loc[i, 'CODI_VARIABLE'] == 35:
#              df.loc[row, 'Rain'] = rain(meteo.loc[i, 'VALOR_LECTURA'])
#            if meteo.loc[i, 'CODI_VARIABLE'] == 30:
#              df.loc[row, 'Wind'] = wind(meteo.loc[i, 'VALOR_LECTURA'])
#          if df.loc[row, 'Date'].minute >= 45 and date_obj2.minute == 30:
#            if meteo.loc[i+3, 'CODI_VARIABLE'] == 32:
#              df.loc[row, 'Temperature [C]'] = meteo.loc[i+3, 'VALOR_LECTURA']
#            if meteo.loc[i+3, 'CODI_VARIABLE'] == 35:
#              df.loc[row, 'Rain'] = rain(meteo.loc[i+3, 'VALOR_LECTURA'])
#            if meteo.loc[i+3, 'CODI_VARIABLE'] == 30:
#              df.loc[row, 'Wind'] = wind(meteo.loc[i+3, 'VALOR_LECTURA'])

#Geolocation information
BCNGeo = 'https://raw.githubusercontent.com/martgnz/bcn-geodata/master/districtes/districtes.geojson'
all_districts = pd.read_json(BCNGeo)
cont_districts = np.zeros(10, int) #There are 10 districts/neighbours in BCN
cont = 0
routes = []

center_lat = 41.390205
center_lon = 2.154007

city_map = folium.Map(location=[center_lat, center_lon], zoom_start=13.5,  prefer_canvas=True) #prefer_canvas increases performance in some cases
for row in df.index:
  if df.loc[row, 'First Latitude'] != "No information" or df.loc[row, 'First Longitude'] != "No information" or df.loc[row, 'Last Latitude'] != "No information" or df.loc[row, 'Last Longitude'] != "No information":
    if df.loc[row, 'Part of the day'] == "Night":
      route = folium.PolyLine(locations = [(df.loc[row, 'First Latitude'], df.loc[row, 'First Longitude']), (midd_lat1[cont], midd_lon1[cont]), (midd_lat2[cont], midd_lon2[cont]), (df.loc[row, 'Last Latitude'], df.loc[row, 'Last Longitude'])], line_opacity = 9, color = 'black')
    elif df.loc[row, 'Part of the day']== "Early morning" or df.loc[row, 'Part of the day']== "Morning" or df.loc[row, 'Part of the day']== "Noon" or df.loc[row, 'Part of the day']== "Afternoon" or df.loc[row, 'Part of the day']== "Evening":
      route = folium.PolyLine(locations = [(df.loc[row, 'First Latitude'], df.loc[row, 'First Longitude']), (midd_lat1[cont], midd_lon1[cont]), (midd_lat2[cont], midd_lon2[cont]), (df.loc[row, 'Last Latitude'], df.loc[row, 'Last Longitude'])], line_opacity = 9, color = 'orange')
    #route = folium.PolyLine(locations = [(df.loc[row, 'First Latitude'], df.loc[row, 'First Longitude']), (midd_lat1[cont], midd_lon1[cont]), (midd_lat2[cont], midd_lon2[cont]), (df.loc[row, 'Last Latitude'], df.loc[row, 'Last Longitude'])], line_opacity = 9, color = 'blue')
    routes.append(route)
    route.add_to(city_map)
    point = Point(df.loc[row, 'First Longitude'], df.loc[row, 'First Latitude'])
    #Haversine formula
    distance = haversine((df.loc[row, 'First Longitude'], df.loc[row, 'First Latitude']), (midd_lon1[cont], midd_lat1[cont]), unit=Unit.METERS) + haversine((midd_lon1[cont], midd_lat1[cont]), (midd_lon2[cont], midd_lat2[cont]), unit=Unit.METERS) + haversine((midd_lon2[cont], midd_lat2[cont]), (df.loc[row, 'Last Longitude'], df.loc[row, 'Last Latitude']), unit=Unit.METERS)
    df.loc[row, 'Distance [m]'] = round(distance, 2)
    df.loc[row, 'Speed [km/h]'] = round(distance*3.6/(df.loc[row, 'Duration [s]']), 2) #In km/h
    cont = cont + 1
    for feature in all_districts['features']:
      polygon = shape(feature['geometry'])
      if polygon.contains(point): #Check in which polygon (district) contains the starting point of the route
        df.loc[row, 'District'] = feature['properties']['NOM'] #Counter for the number of routes in each district
        if feature['properties']['NOM'] == 'Eixample':
          cont_districts[0] += 1
        if feature['properties']['NOM'] == 'Ciutat Vella':
          cont_districts[1] += 1
        if feature['properties']['NOM'] == 'Sants-Montjuïc':
          cont_districts[2] += 1
        if feature['properties']['NOM'] == 'Les Corts':
          cont_districts[3] += 1
        if feature['properties']['NOM'] == 'Sarrià-Sant Gervasi':
          cont_districts[4] += 1
        if feature['properties']['NOM'] == 'Gràcia':
          cont_districts[5] += 1
        if feature['properties']['NOM'] == 'Horta-Guinardó':
          cont_districts[6] += 1
        if feature['properties']['NOM'] == 'Nou Barris':
          cont_districts[7] += 1
        if feature['properties']['NOM'] == 'Sant Andreu':
          cont_districts[8] += 1
        if feature['properties']['NOM'] == 'Sant Martí':
          cont_districts[9] += 1
  else:
    routes.append("No information")

#df.to_csv('database_2022.csv', index=None, columns=None)
folium.TileLayer('cartodbpositron').add_to(city_map)

#Visualize the map
data2 = []

city_graph_2 = ox.graph_from_place(['Barcelona, Barcelona, Spain'], network_type = 'all_private')

for i in cont_districts:
  data2.append(i)

df2 = pd.DataFrame(data2, index=['Eixample', 'Ciutat Vella', 'Sants-Montjuïc', 'Les Corts', 'Sarrià-Sant Gervasi', 'Gràcia', 'Horta-Guinardó', 'Nou Barris', 'Sant Andreu', 'Sant Martí'], columns=['Number of routes'])

#city_map.choropleth(geo_data = BCNGeo, data=df2, 
#                    columns=[df2.index, 'Number of routes'], key_on='feature.properties.NOM', 
#                    fill_color='BuPu', fill_opacity=0.5, line_opacity=0.4, legend_name='Number of routes per district', smooth_factor=0, bins=5)

# We specify a lambda function mapping a GeoJson Feature to a style dict
#style_function = lambda x: {'fillColor': '#eef4ff', 
#                            'color':'#eef4ff', 
#                            'fillOpacity': 0.1, 
#                            'weight': 0.1}

# We specify a function mapping a GeoJson Feature to a style dict for mouse events, in this case "highlighting"
#highlight_function = lambda x: {'fillColor': '#eef4ff', 
#                                'color':'#eef4ff', 
#                                'fillOpacity': 0.70, 
#                                'weight': 0.1}

# We create a new layer for the map which is going to give us the interactivity
#BCNT = folium.features.GeoJson(BCNGeo, style_function=style_function,
#    control=False,
#    highlight_function=highlight_function, 
#    tooltip=folium.features.GeoJsonTooltip(fields=['NOM'], aliases=['District name:'], style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")))

#city_map.add_child(BCNT) # We add this new layer
#city_map.keep_in_front(BCNT) # By keeping it in front we will ensure that each time we deploy the map, this layer will be in the front
#folium.LayerControl().add_to(city_map)
folium.TileLayer('cartodbpositron').add_to(city_map)
city_map.save('map_2022_daynight.html')