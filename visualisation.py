#%%

import pandas as pd
from download import download

# to get the current working dir and change it
import os

# to create the animation
import plotly.express as px
import plotly.graph_objects as go

#%% changing the cwd

os.chdir(str(os.getcwd()))

#%% data treatment
import os
import glob
import pandas as pd

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combinedFile = combined_csv.to_csv( "../combined_csv.csv", index=False, encoding='utf-8-sig')

#fichier noms compteurs
names = pd.read_csv("../noms_compteurs.csv")
names.columns = [ "name", "serialNumber", "1", "2", "3", "id"]
names['name'] = ['Tanneurs', 'Berracasa', 'Celleneuve', 'Lavérune', 'Vieille Poste', 'Delmas 1', 'Delmas 2', 'Gerhardt', 'Lattes 2', 'Lattes 1']

#%% downloading datas
url=""
os.mkdir("./csv")
path_target = "./csv/allDatas.csv" 
download(url, path_target, replace=True) #downloads the file to path_target

#%%
df = pd.read_csv("../allDatas.csv")
df = df.drop(columns = ["location/type", "id", "type", "vehicleType", "reversedLane"])
df.columns = ["intensity", "laneId", "date", "lat", "long"]
nom = names[['name', 'id']]
df['name'] = [nom[nom['id'] == df['laneId'][k]]['name'].iloc[0] for k in range(len(df))]
df["date"] = [df["date"][k][:10] for k in range(len(df))]
df = df.sort_values(['date'])

#%%

fig = px.scatter_geo(df,
                    lon = df['lat'], 
                    lat = df['long'], 
                    size="intensity",
                    animation_frame = "date",
                    animation_group = 'laneId', 
                    hover_name = 'name',
                    hover_data=list(['intensity', 'date']),
                    color = 'intensity')

# focus point on MPL
lat_foc = 43.610769
lon_foc = 3.876716

fig.update_geos(fitbounds="locations", visible=True, showland = True)

#fig = fig.update_layout(mapbox_style="open-street-map")
#fig.update_layout_images(visible=True, source = 'https://www.google.com/maps/place/France/@45.8665231,-6.9240942') 
#fig.update_geos(showland = True)

#
fig.write_html("./file2.html")
fig.show()
# %%
import plotly.graph_objects as go

fig = go.Figure(go.Scattermapbox(
    mode = "markers",
    marker = {'size': 20, 'color': ["cyan"]}))

fig.update_layout(
    mapbox = {
        'style': "stamen-terrain",
        'zoom': 12, 'layers': [{
            'source': {
                'type': "FeatureCollection",
                'features': [{
                    'type': "Feature",
                    'geometry': {
                        'type': "MultiPolygon",
                    }
                }]
            },
            'type': "fill", 'below': "traces", 'color': "royalblue"}]},
    margin = {'l':0, 'r':0, 'b':0, 't':0})

fig.show()
# %%
import plotly.express as px
df = px.data.gapminder()
fig = px.scatter_geo(df, locations="iso_alpha", color="continent",
                     hover_name="country", size="pop",
                     animation_frame="year",
                     projection="equirectangular")
fig = fig.update_layout(mapbox_style="open-street-map")
fig.show()
# %%
fig = px.scatter_mapbox(df, lat="long", 
                    lon="lat", 
                    hover_name="name", 
                    animation_frame = "date",
                    animation_group = 'laneId',
                    hover_data=["intensity", "date"],
                    color = 'intensity', 
                    size = 'intensity',
                    height = 500, 
                    zoom = 10)

                      

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.update_layout(mapbox_style="open-street-map")
#fig.update_geos(fitbounds="locations", visible=True, showland = True)
fig.write_html("./file2.html")

fig.show()
# %%
