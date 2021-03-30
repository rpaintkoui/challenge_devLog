#%%
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#import cartopy.crs as ccrs

import pandas as pd



# %%

df = pd.read_csv(‘data_Jour.csv’, names=[‘School’, ‘Grad Date’, ‘Long’, ‘Lat’])
# convert the graduation date column to datetime objects
df[‘Grad Date’] = pd.to_datetime(df[‘Grad Date’])
#%%
fig = plt.figure(figsize=(19.2, 10.8))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=0,  
                                       min_latitude=-65,
                                       max_latitude=70))
#%%
fig = plt.figure(figsize=(19.2, 10.8))
ax = plt.axes(projection=ccrs.Mercator(central_longitude=0,  
                                       min_latitude=-65,
                                       max_latitude=70))
# %%
date = datetime(2017, 12, 31)
grads = df[df['Grad Date'] <= date]
# %%
# Define colors for each school
colors = {'AI': '#02b3e4',
          'Aut Sys': '#f95c3c' ,
          'Business': '#ff5483',
          'Developers': '#ecc81a'}
for school, school_data in grads.groupby('School'):
    
    grad_counts = school_data.groupby(['Long', 'Lat']).count()
    
    # Get lists for longitudes and latitudes of graduates
    index = list(grad_counts.index)
    longs = [each[0] for each in index]
    lats = [each[1] for each in index]
    sizes = grad_counts['School']*10
    # The school names are like 'School of AI', remove 'School of '
    school_name = ' '.join(school.split()[2:])
    
    ax.scatter(longs, lats, s=sizes,
               color=colors[school_name], alpha=0.8,
               transform=ccrs.PlateCarree())
# %%
fontname = 'Open Sans'
fontsize = 28
# Positions for the date and grad counter
date_x = -53
date_y = -50
date_spacing = 65
# Positions for the school labels
name_x = -70
name_y = -60      
name_spacing = {'Developers': 0,
                'AI': 55,
                'Business': 1.9*55,
                'Aut Sys': 3*55}
# Date text
ax.text(date_x, date_y, 
        f"{date.strftime('%b %d, %Y')}", 
        color='white',
        fontname=fontname, fontsize=fontsize*1.3,
        transform=ccrs.PlateCarree())
# Total grad counts
ax.text(date_x + date_spacing, date_y, 
        "GRADUATES", color='white',
        fontname=fontname, fontsize=fontsize,
        transform=ccrs.PlateCarree())
ax.text(date_x + date_spacing*1.7, date_y, 
        f"{grads.groupby(['Long', 'Lat']).count()['School'].sum()}",
        color='white', ha='left',
        fontname=fontname, fontsize=fontsize*1.3,
        transform=ccrs.PlateCarree())
for school_name in ['Developers', 'AI', 'Business', 'Aut Sys']:
    ax.text(name_x + name_spacing[school_name], 
            name_y, 
            school_name.upper(), ha='center',
            fontname=fontname, fontsize=fontsize*1.1,
            color=colors[school_name],
            transform=ccrs.PlateCarree())
# Expands image to fill the figure and cut off margins
fig.tight_layout(pad=-0.5)
# %%
def make_grads_map(date, data, ax=None, resolution='low'):
    
    if ax is None:
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = plt.axes(projection=ccrs.Mercator(min_latitude=-65,
                                               max_latitude=70))
    
    ax.background_img(name='BM', resolution=resolution)
    ax.set_extent([-170, 179, -65, 70], crs=ccrs.PlateCarree())
    grads = data[data['Grad Date'] < date] 
    
    ### rest of the code
start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 3, 15)
fig = plt.figure(figsize=(19.2, 10.8))
ax = plt.axes(projection=ccrs.Mercator(min_latitude=-65,
                                       max_latitude=70))
# Generate an image for each day between start_date and end_date
for ii, days in enumerate(range((end_date - start_date).days)):
    date = start_date + timedelta(days)
    ax = make_grads_map(date, df, ax=ax, resolution='full')
    fig.tight_layout(pad=-0.5)
    fig.savefig(f"frames/frame_{ii:04d}.png", dpi=100,     
                frameon=False, facecolor='black')
    ax.clear()
# %%

def make_grads_map(date, data, ax=None, resolution='low'):
    
    if ax is None:
        fig = plt.figure(figsize=(19.2, 10.8))
        ax = plt.axes(projection=ccrs.Mercator(min_latitude=-65,
                                               max_latitude=70))
    
    ax.background_img(name='BM', resolution=resolution)
    ax.set_extent([-170, 179, -65, 70], crs=ccrs.PlateCarree())
    grads = data[data['Grad Date'] < date] 
    
    ### rest of the code
start_date = datetime(2017, 1, 1)
end_date = datetime(2018, 3, 15)
fig = plt.figure(figsize=(19.2, 10.8))
ax = plt.axes(projection=ccrs.Mercator(min_latitude=-65,
                                       max_latitude=70))
# Generate an image for each day between start_date and end_date
for ii, days in enumerate(range((end_date - start_date).days)):
    date = start_date + timedelta(days)
    ax = make_grads_map(date, df, ax=ax, resolution='full')
    fig.tight_layout(pad=-0.5)
    fig.savefig(f"frames/frame_{ii:04d}.png", dpi=100,     
                frameon=False, facecolor='black')
    ax.clear()
