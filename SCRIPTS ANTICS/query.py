import folium             # Visualization library  
from shapely.geometry import MultiPoint, Point, Polygon, shape   # Handle geometry
import pandas as pd
from database import routes

df = pd.read_csv('table_df.csv')
df_query = df.query("`Part of the day`=='Morning'") 
#District=='Eixample'
#`Duration [s]` >= 16
rows = df_query.index
df_query.to_csv('query_df.csv', index=None, columns=None)

center_lat = 41.390205
center_lon = 2.154007

city_map_query = folium.Map(location=[center_lat, center_lon], zoom_start=13.5,  prefer_canvas=True) #prefer_canvas increases performance in some cases

for row in df_query.index:
  if df.loc[row, 'First Latitude'] != "No information" or df.loc[row, 'First Longitude'] != "No information" or df.loc[row, 'Last Latitude'] != "No information" or df.loc[row, 'Last Longitude'] != "No information":
    if routes[row]!="No information":
      routes[row].add_to(city_map_query)

folium.TileLayer('cartodbpositron').add_to(city_map_query)
city_map_query.save('city_map_query.html')