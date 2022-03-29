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

root = "/home/usuaris/imatge/mireia.ambros/videos/"
data = []
midd_lat1 = []
midd_lon1 = []
midd_lat2 = []
midd_lon2 = []

for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in sorted(os.listdir(root + folder)):
          if file.find("mp4")!=-1:
            info = ffmpeg.probe(root + folder + "/" + file)
            video_info = next(stream for stream in info['streams'] if stream['codec_type'] == 'video') #Extract metadata using ffmpeg
            time = (video_info['duration'])
            num_frame = int(video_info['nb_frames'])
            date = (video_info['tags']['creation_time'])
            date = date.replace('T', ' ')
            date = date.replace('.000000Z', '') #Date format YYYY-MM-DD (HH:MM:SS), Zulu time at the zero meridian
            date_obj = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') #Convert the date to a datetime object
            timezone = pytz.utc
            date_utc = timezone.localize(date_obj)
            date_local = date_utc.astimezone(pytz.timezone("Europe/Madrid")) #Shows the local time and the correction made to the Zulu time zone
            data.append((folder, file, root + folder, date_local, round(float(time), 2), round(num_frame/float(time), 2), "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information", "No information"))     
        df = pd.DataFrame(data, columns=['Road type', 'Filename', 'Path', 'Date', 'Duration [s]', 'Resolution [fps]', 'First Latitude', 'First Longitude', 'Last Latitude', 'Last Longitude', 'District', 'Distance [m]', 'Speed [km/h]', 'Part of the day', 'Temperature [ºC]', 'Rain', 'Wind'])
        
for folder in os.listdir(root):
    if os.path.isdir(os.path.join(root, folder)):
        for file in sorted(os.listdir(root + folder)):
          if file.find("idx")!=-1:
            filename = file.replace('.idx', '.')
            this_row = df.index[df['Filename'] == filename + "mp4"].tolist() #In which row (file) the location should be inserted
            # Get first and last coordinates of idx file
            with open(root + folder + "/" + file, "r") as f:
              line = f.readlines()
              first_lat = line[0].split(' ')[4]
              first_lon = line[0].split(' ')[5]
              last_lat = line[-1].split(' ')[4]
              last_lon = line[-1].split(' ')[5]
              midd_lat1.append(float(line[-300].split(' ')[4]))
              midd_lon1.append(float(line[-300].split(' ')[5]))
              midd_lat2.append(float(line[-150].split(' ')[4]))
              midd_lon2.append(float(line[-150].split(' ')[5]))
            df.iloc[this_row, 6] = round(float(first_lat), 6)
            df.iloc[this_row, 7] = round(float(first_lon), 6) 
            df.iloc[this_row, 8] = round(float(last_lat), 6) 
            df.iloc[this_row, 9] = round(float(last_lon), 6)

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
    route = folium.PolyLine(locations = [(df.loc[row, 'First Latitude'], df.loc[row, 'First Longitude']), (midd_lat1[cont], midd_lon1[cont]), (midd_lat2[cont], midd_lon2[cont]), (df.loc[row, 'Last Latitude'], df.loc[row, 'Last Longitude'])], line_opacity = 9, color = 'red')
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
          cont_districts[0] = cont_districts[0] + 1
        if feature['properties']['NOM'] == 'Ciutat Vella':
          cont_districts[1] = cont_districts[1] + 1
        if feature['properties']['NOM'] == 'Sants-Montjuïc':
          cont_districts[2] = cont_districts[2] + 1
        if feature['properties']['NOM'] == 'Les Corts':
          cont_districts[3] = cont_districts[3] + 1
        if feature['properties']['NOM'] == 'Sarrià-Sant Gervasi':
          cont_districts[4] = cont_districts[4] + 1
        if feature['properties']['NOM'] == 'Gràcia':
          cont_districts[5] = cont_districts[5] + 1
        if feature['properties']['NOM'] == 'Horta-Guinardó':
          cont_districts[6] = cont_districts[6] + 1
        if feature['properties']['NOM'] == 'Nou Barris':
          cont_districts[7] = cont_districts[7] + 1
        if feature['properties']['NOM'] == 'Sant Andreu':
          cont_districts[8] = cont_districts[8] + 1
        if feature['properties']['NOM'] == 'Sant Martí':
          cont_districts[9] = cont_districts[9] + 1
  else:
    routes.append("No information")

#Weather and part of the day information
meteo = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Dades_meteorologiques_de_la_XEMA.csv", sep = ',')

sun_2018 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2018.csv")
sun_2019 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2019.csv")
sun_2010 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2020.csv")
sun_2021 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2021.csv")
sun_2022 = pd.read_csv("/home/usuaris/imatge/mireia.ambros/meteo/Barcelona-sun-2022.csv")

for row in df.index:
  if df.loc[row, 'Date'].year == 2018:
    info_year = sun_2018
  if df.loc[row, 'Date'].year == 2019:
    info_year = sun_2019
  if df.loc[row, 'Date'].year == 2020:
    info_year = sun_2010
  if df.loc[row, 'Date'].year == 2021:
    info_year = sun_2021
  if df.loc[row, 'Date'].year == 2022:
    info_year = sun_2022
  for n in info_year.index:
    date_obj3 = datetime.datetime.strptime(sun_2021.loc[n, 'Fecha'], '%d/%m/%Y')
    if date_obj3.day == df.loc[row, 'Date'].day and date_obj3.month == df.loc[row, 'Date'].month and date_obj3.year == df.loc[row, 'Date'].year:
      date_obj4 = datetime.datetime.strptime(sun_2021.loc[n, 'Salida de sol'], '%I:%M:%S %p')
      date_obj5 = datetime.datetime.strptime(sun_2021.loc[n, 'Puesta de sol'], '%I:%M:%S %p')
      date_obj6 = datetime.datetime.strptime("9:00:59", '%H:%M:%S')
      date_obj7 = datetime.datetime.strptime("9:01:00", '%H:%M:%S')
      date_obj8 = datetime.datetime.strptime("12:00:59", '%H:%M:%S')
      date_obj9 = datetime.datetime.strptime("12:01:00", '%H:%M:%S')
      date_obj10 = datetime.datetime.strptime("17:00:59", '%H:%M:%S')
      date_obj11 = datetime.datetime.strptime("17:01:00", '%H:%M:%S')
      if df.loc[row, 'Date'].time() >= date_obj4.time() and df.loc[row, 'Date'].time() <= date_obj6.time():
          df.loc[row, 'Part of the day'] = "Early morning"
      if df.loc[row, 'Date'].time() >= date_obj7.time() and df.loc[row, 'Date'].time() <= date_obj8.time():
          df.loc[row, 'Part of the day'] = "Morning"
      if df.loc[row, 'Date'].time() >= date_obj9.time() and df.loc[row, 'Date'].time() <= date_obj10.time():
          df.loc[row, 'Part of the day'] = "Afternoon"
      if df.loc[row, 'Date'].time() >= date_obj11.time() and df.loc[row, 'Date'].time() <= date_obj5.time():
          df.loc[row, 'Part of the day'] = "Evening"
      if df.loc[row, 'Date'].time() < date_obj4.time() and df.loc[row, 'Date'].time() > date_obj5.time():
        df.loc[row, 'Part of the day'] = "Night"
  for i in meteo.index:
    date_found = False
    date_obj2 = datetime.datetime.strptime(meteo.loc[i, 'DATA_LECTURA'], '%d/%m/%Y %I:%M:%S %p')
    if date_obj2.day == df.loc[row, 'Date'].day and date_obj2.month == df.loc[row, 'Date'].month and date_obj2.year == df.loc[row, 'Date'].year and date_found == False:
      date_found =  True
      if date_obj2.hour == df.loc[row, 'Date'].hour:
        if meteo.loc[i, 'DATA_LECTURA'].find("AM")!=-1:
          info_hour = "AM"
        if meteo.loc[i, 'DATA_LECTURA'].find("PM")!=-1:
          info_hour = "PM"
        if meteo.loc[i, 'DATA_LECTURA'].find(info_hour)!=-1:
          if df.loc[row, 'Date'].minute < 15 and date_obj2.minute == 0:
            if meteo.loc[i, 'CODI_VARIABLE'] == 32:
              df.loc[row, 'Temperature [ºC]'] = meteo.loc[i, 'VALOR_LECTURA']
            if meteo.loc[i, 'CODI_VARIABLE'] == 35:
              if meteo.loc[i, 'VALOR_LECTURA'] == 0:
                df.loc[row, 'Rain'] = "No rain"
              if meteo.loc[i, 'VALOR_LECTURA'] > 0 and meteo.loc[i, 'VALOR_LECTURA'] <= 2:
                df.loc[row, 'Rain'] = "Weak"
              if meteo.loc[i, 'VALOR_LECTURA'] > 2 and meteo.loc[i, 'VALOR_LECTURA'] <= 15:
                df.loc[row, 'Rain'] = "Moderate"
              if meteo.loc[i, 'VALOR_LECTURA'] > 15 and meteo.loc[i, 'VALOR_LECTURA'] <= 30:
                df.loc[row, 'Rain'] = "Strong"
              if meteo.loc[i, 'VALOR_LECTURA'] > 30 and meteo.loc[i, 'VALOR_LECTURA'] <= 60:
                df.loc[row, 'Rain'] = "Very heavy"
              if meteo.loc[i, 'VALOR_LECTURA'] > 60:
                df.loc[row, 'Rain'] = "Torrential"
            if meteo.loc[i, 'CODI_VARIABLE'] == 30:
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) < 15:
                df.loc[row, 'Wind'] = "No wind"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) >= 15 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 40:
                df.loc[row, 'Wind'] = "Moderate"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 40 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 70:
                df.loc[row, 'Wind'] = "Strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 70 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 120:
                df.loc[row, 'Wind'] = "Very strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 120:
                df.loc[row, 'Wind'] = "Hurricanes"
          if df.loc[row, 'Date'].minute >=15 and df.loc[row, 'Date'].minute < 45 and date_obj2.minute == 30:
            if meteo.loc[i, 'CODI_VARIABLE'] == 32:
              df.loc[row, 'Temperature [ºC]'] = meteo.loc[i, 'VALOR_LECTURA']
            if meteo.loc[i, 'CODI_VARIABLE'] == 35:
              if meteo.loc[i, 'VALOR_LECTURA'] == 0:
                df.loc[row, 'Rain'] = "No rain"
              if meteo.loc[i, 'VALOR_LECTURA'] > 0 and meteo.loc[i, 'VALOR_LECTURA'] <= 2:
                df.loc[row, 'Rain'] = "Weak"
              if meteo.loc[i, 'VALOR_LECTURA'] > 2 and meteo.loc[i, 'VALOR_LECTURA'] <= 15:
                df.loc[row, 'Rain'] = "Moderate"
              if meteo.loc[i, 'VALOR_LECTURA'] > 15 and meteo.loc[i, 'VALOR_LECTURA'] <= 30:
                df.loc[row, 'Rain'] = "Strong"
              if meteo.loc[i, 'VALOR_LECTURA'] > 30 and meteo.loc[i, 'VALOR_LECTURA'] <= 60:
                df.loc[row, 'Rain'] = "Very heavy"
              if meteo.loc[i, 'VALOR_LECTURA'] > 60:
                df.loc[row, 'Rain'] = "Torrential"
            if meteo.loc[i, 'CODI_VARIABLE'] == 30:
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) < 15:
                df.loc[row, 'Wind'] = "No wind"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) >= 15 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 40:
                df.loc[row, 'Wind'] = "Moderate"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 40 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 70:
                df.loc[row, 'Wind'] = "Strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 70 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 120:
                df.loc[row, 'Wind'] = "Very strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 120:
                df.loc[row, 'Wind'] = "Hurricanes"
          if df.loc[row, 'Date'].minute > 45 and date_obj2.minute == 30:
            if meteo.loc[i+4, 'CODI_VARIABLE'] == 32:
              df.loc[row, 'Temperature [ºC]'] = meteo.loc[i+4, 'VALOR_LECTURA']
            if meteo.loc[i, 'CODI_VARIABLE'] == 35:
              if meteo.loc[i, 'VALOR_LECTURA'] == 0:
                df.loc[row, 'Rain'] = "No rain"
              if meteo.loc[i, 'VALOR_LECTURA'] > 0 and meteo.loc[i, 'VALOR_LECTURA'] <= 2:
                df.loc[row, 'Rain'] = "Weak"
              if meteo.loc[i, 'VALOR_LECTURA'] > 2 and meteo.loc[i, 'VALOR_LECTURA'] <= 15:
                df.loc[row, 'Rain'] = "Moderate"
              if meteo.loc[i, 'VALOR_LECTURA'] > 15 and meteo.loc[i, 'VALOR_LECTURA'] <= 30:
                df.loc[row, 'Rain'] = "Strong"
              if meteo.loc[i, 'VALOR_LECTURA'] > 30 and meteo.loc[i, 'VALOR_LECTURA'] <= 60:
                df.loc[row, 'Rain'] = "Very heavy"
              if meteo.loc[i, 'VALOR_LECTURA'] > 60:
                df.loc[row, 'Rain'] = "Torrential"
            if meteo.loc[i, 'CODI_VARIABLE'] == 30:
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) < 15:
                df.loc[row, 'Wind'] = "No wind"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) >= 15 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 40:
                df.loc[row, 'Wind'] = "Moderate"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 40 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 70:
                df.loc[row, 'Wind'] = "Strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 70 and (meteo.loc[i, 'VALOR_LECTURA']*3.6) <= 120:
                df.loc[row, 'Wind'] = "Very strong"
              if (meteo.loc[i, 'VALOR_LECTURA']*3.6) > 120:
                df.loc[row, 'Wind'] = "Hurricanes"

df.to_csv('table_df.csv', index=None, columns=None)
folium.TileLayer('cartodbpositron').add_to(city_map)

#Visualize the map
data2 = []

city_graph_2 = ox.graph_from_place(['Barcelona, Barcelona, Spain'], network_type = 'all_private')

for i in cont_districts:
  data2.append(i)

df2 = pd.DataFrame(data2, index=['Eixample', 'Ciutat Vella', 'Sants-Montjuïc', 'Les Corts', 'Sarrià-Sant Gervasi', 'Gràcia', 'Horta-Guinardó', 'Nou Barris', 'Sant Andreu', 'Sant Martí'], columns=['Number of routes'])

city_map.choropleth(geo_data = BCNGeo, data=df2, 
                    columns=[df2.index, 'Number of routes'], key_on='feature.properties.NOM', 
                    fill_color='YlOrRd', fill_opacity=0.2, line_opacity=0.4, legend_name='Number of routes per district', smooth_factor=0, bins=5)

# We specify a lambda function mapping a GeoJson Feature to a style dict
style_function = lambda x: {'fillColor': '#eef4ff', 
                            'color':'#eef4ff', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}

# We specify a function mapping a GeoJson Feature to a style dict for mouse events, in this case "highlighting"
highlight_function = lambda x: {'fillColor': '#eef4ff', 
                                'color':'#eef4ff', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

# We create a new layer for the map which is going to give us the interactivity
BCNT = folium.features.GeoJson(BCNGeo, style_function=style_function,
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(fields=['NOM'], aliases=['District name:'], style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")))

city_map.add_child(BCNT) # We add this new layer
city_map.keep_in_front(BCNT) # By keeping it in front we will ensure that each time we deploy the map, this layer will be in the front
folium.LayerControl().add_to(city_map)
folium.TileLayer('cartodbpositron').add_to(city_map)
city_map.save('city_map.html')