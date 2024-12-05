# %%
import geoplot as gplt # Geoplot: Library for high-level geospatial data visualization
import geopandas as gpd # GeoPandas: Extends pandas to handle geometric/geographic data
import geoplot.crs as gcrs # Geoplot color ramp schemes for geographic visualizations
import imageio # Imageio: Library for reading and writing image data
import pandas as pd # Pandas: Data manipulation and analysis library
import pathlib # Pathlib: Object-oriented filesystem paths
import matplotlib.animation as animation # Matplotlib animation: For creating animated visualizations
import matplotlib.pyplot as plt # Matplotlib pyplot: Plotting library for creating static visualizations
import mapclassify as mc # Mapclassify: Classification schemes for choropleth maps
import numpy as np # NumPy: Library for numerical computing in Python
import pycountry # Pycountry: Database of country information (names, ISO codes, etc.)
import plotly.express as px # Plotly Express: High-level interface for creating interactive plots
import seaborn as sns
import pandas as pd
import pycountry
import plotly.express as px

# %%
usa = gpd.read_file("C:\\Users\\emmae\\Desktop\\DataVisualization_Uppgiftsmapp\\Labb2\\3.3NationalObesityByState\\us_state.shp")
print(usa.head())

# %%
state_pop = pd.read_csv("C:\\Users\\emmae\\Desktop\\DataVisualization_Uppgiftsmapp\\Labb2\\3.3GeospatialDataVisualization\\us_state_est_population.csv")
print(state_pop.head())

# %%
#Couples state name and region name
pop_states = usa.merge(state_pop, left_on="StateName", right_on="NAME")
pop_states.head()

# %%
path = gplt.datasets.get_path("contiguous_usa")
contiguous_usa = gpd.read_file(path)
contiguous_usa.head()

# %% [markdown]
# **4.5.2.13 Another sample dataset, in this case for US cities**

# %%
path = gplt.datasets.get_path("usa_cities")
usa_cities = gpd.read_file(path)
usa_cities.head()

# %%
continental_usa_cities = usa_cities.query('STATE not in ["HI", "AK", "PR"]')
gplt.pointplot(continental_usa_cities, s=3, figsize=(20,15))

# %%
ax = gplt.polyplot(contiguous_usa)
gplt.pointplot(continental_usa_cities, s=1, ax=ax)

# %%
ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())
gplt.pointplot(continental_usa_cities, s= 1, ax=ax)

# %%
# Define dimensions of the grid
rows = 10
columns = 15

# Create a 2D coordinate grid using meshgrid
x, y = np.meshgrid(np.linspace(-1,1,columns), np.linspace(-1,1,rows))

# Calculate the distance from origin (0,0) for each point using Pythagorean theorem
d = np.sqrt(x*x+y*y)

# Define standard deviation for the Gaussian distribution
sigma = 0.5

# Create a 2D Gaussian disc:
disc = (8*np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ))).astype('uint')

# Create an RGB color array by stacking three versions of the disc:
# Original disc for red channel
# Rolled (shifted) disc by 2 positions along axis 0 for green channel
# Rolled disc by 2 positions along axis 1 for blue channel
# axis=2 means we're stacking these as color channels (R,G,B)
myRGBColorArray = np.stack((disc,np.roll(disc,2,axis=0),np.roll(disc,2,axis=1)),axis=2)

# Print and display the Green channel (index 1) of the RGB array
print("Green:")
plt.imshow(myRGBColorArray[:,:,1], cmap='Greens')
plt.show()

# %%
ax = gplt.polyplot(
    contiguous_usa,
    edgecolor="white",
    facecolor="lightgray",
    figsize=(12, 8),
    projection=gcrs.AlbersEqualArea()
)

gplt.pointplot(
    continental_usa_cities,
    ax=ax,
    hue="ELEV_IN_FT",
    cmap="Blues",
    scheme="quantiles",
    scale="ELEV_IN_FT",
    limits=(1, 10),
    legend=True,
    legend_var="scale",
    legend_kwargs={"frameon": False},
    legend_values=[-110, 1750, 3600, 5500, 7400],
    legend_labels=["-110 feet", "1750 feet", "3600 feet", "5500 feet", "7400 feet"]
)

ax.set_title("Cities in the continental US, by elevation", fontsize=16)

# %%
ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())
gplt.choropleth(
    contiguous_usa,
    hue="population",
    edgecolor="white",
    linewidth=1,
    cmap="Greens",
    legend=True,
    scheme="FisherJenks",
    legend_labels=[
        "<3 million", "3-6.7 million", "6.7-12.8 million",
        "12.8-25 million", "25-37 million"
    ],
    projection=gcrs.AlbersEqualArea(),
    ax=ax
)

# %%
df_confirmedGlobal = pd.read_csv("C:\\Users\\emmae\\Desktop\\DataVisualization_Uppgiftsmapp\\Labb2\\3.4Covid-19\\time_series_covid19_confirmed_global.csv")
print(df_confirmedGlobal.head())

# %%
# Remove unnecessary columns and sum by country
df_confirmedGlobal = df_confirmedGlobal.drop(columns=['Province/State', 'Lat', 'Long'])
df_confirmedGlobal = df_confirmedGlobal.groupby('Country/Region').agg('sum')
date_list = list(df_confirmedGlobal.columns)

# Function to get three-letter country codes
def get_country_code(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None

# Add country codes to the dataset
df_confirmedGlobal['country'] = df_confirmedGlobal.index
df_confirmedGlobal['iso_alpha_3'] = df_confirmedGlobal['country'].apply(get_country_code)

# Transform to long format
df_long = pd.melt(df_confirmedGlobal, 
                  id_vars=['country', 'iso_alpha_3'], 
                  value_vars=date_list)
print(df_long)

# %%
fig = px.choropleth(df_long,
                    locations="iso_alpha_3",
                    color="value",
                    hover_name="country",
                    animation_frame="variable",
                    projection="natural earth",
                    color_continuous_scale='Peach',
                    range_color=[0, 50000]
                    )

fig.show()
fig.write_html("Covid19_map.html") # write it to HTML file

# %% [markdown]
# # 5 Part#3: Challenging Questions

# %% [markdown]
# **Q1. Use “myRGBColorArray” from step5 in subsection Visualizing Spatial Data and
# display individual channels as well as Composite accordingly.**

# %%
# Create a figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Display the Red channel
axes[0].imshow(myRGBColorArray[:, :, 0], cmap='Reds')
axes[0].set_title('Red Channel')
axes[0].axis('off')

# Display the Green channel
axes[1].imshow(myRGBColorArray[:, :, 1], cmap='Greens')
axes[1].set_title('Green Channel')
axes[1].axis('off')

# Display the Blue channel
axes[2].imshow(myRGBColorArray[:, :, 2], cmap='Blues')
axes[2].set_title('Blue Channel')
axes[2].axis('off')

# Display the Composite image
axes[3].imshow(myRGBColorArray)
axes[3].set_title('Composite Image')
axes[3].axis('off')


plt.show()

# %%
rows = 10
columns = 15
x, y = np.meshgrid(np.linspace(-1, 1, columns), np.linspace(-1, 1, rows))
sigma = 0.5
d = np.sqrt(x**2 + y**2)
disc = (8 * np.exp(-(d**2 / (2.0 * sigma**2)))).astype('float32')  # Använd float för mer kontroll

#Normalisera disken mellan 0 och 255
disc = 255 * (disc / disc.max())

#Skapa RGB-array
myRGBColorArray = np.stack([
    disc,                     # Röd kanal
    np.roll(disc, 2, axis=0), # Grön kanal
    np.roll(disc, 2, axis=1)  # Blå kanal
], axis=2).astype('uint8')

#Visa bilderna
plt.figure(figsize=(10, 3))

#Röd kanal
plt.subplot(1, 4, 1)
plt.title("Red Channel")
plt.imshow(myRGBColorArray[:, :, 0], cmap='Reds')
plt.axis('off')

#Grön kanal
plt.subplot(1, 4, 2)
plt.title("Green Channel")
plt.imshow(myRGBColorArray[:, :, 1], cmap='Greens')
plt.axis('off')

#Blå kanal
plt.subplot(1, 4, 3)
plt.title("Blue Channel")
plt.imshow(myRGBColorArray[:, :, 2], cmap='Blues')
plt.axis('off')

#Kombinerad RGB-bild som blir helsvart
plt.subplot(1, 4, 4)
plt.title("Composite RGB colorful")
plt.imshow(myRGBColorArray)
plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# The first three subplots display when the emphasis is on one out of the three possible channels, either red, green or blue. The fourth subplot, the composite subplot, is made from all three channels. In our first code the composite subplot was all blanck and in the second it was more colorful. We added the second composite subplot in order to make it more esthetically appealing. 

# %% [markdown]
# **Q2. What is quantile? How does the number of quantiles affect the visualization? (write
# 3-5 sentences)**

# %% [markdown]
# A quantile divides a dataset into equal-sized subsets based on the data's distribution. The number of quantiles determines how finely the data is partitioned and affect how patterns/distributions are visualized. When fewer quantiles are used (e.g., quartiles), the visualization highlights broader trends, making it easier to observe general patterns but not finer details. Conversely, using many quantiles reveals more nuanced or subtle variations in the data, but can make the visualization cluttered and harder to interpret. By adjusting the number of quantiles, you balance between simplicity and detail in your visualizations.

# %% [markdown]
# **Q3. Divide the continental_usa_cities into 10 quantiles and assign a different hue to each
# quantile and a legend to explain accordingly. Produce a point plot or similar plot and
# explain the representation difference in your own words.**

# %%
#import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Classify the data into 10 quantiles
quantiles = mc.Quantiles(continental_usa_cities['ELEV_IN_FT'], k=10)

# Add the quantile classification to the dataframe
continental_usa_cities['quantiles'] = quantiles.yb

# Create the plot
ax = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea())
gplt.pointplot(
    continental_usa_cities,
    ax=ax,
    hue='quantiles',
    cmap='viridis',
    legend=False,
    #legend_var='hue',
    s=1
)

# Create a custom legend
cmap = plt.cm.viridis
norm = mcolors.Normalize(vmin=0, vmax=9)  # Quantiles range from 0 to 9
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Required for the colorbar to function
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.05)
# Add the title to the colorbar (acts as the legend heading)
cbar.set_label("Elevation in feet")

plt.title("US Cities Elevation Quantiles")
plt.show()

# %% [markdown]
# Representation Difference: By dividing the data into quantiles, we can clearly see how the cities are distributed across different elevation ranges. Each quantile represents a subset of the data with similar elevations, allowing us to understand the distribution and density of cities at various elevation levels. This method is effective in highlighting patterns and trends that might be missed if we were to use a single color for all data points or group them without considering the distribution. Based on the output we can conclude that Central US is higher above sea level than the coasts. Western part of the US is more elevated than the eastern part of the ocuntry. 

# %% [markdown]
# **Q4. Create a Voronoi diagram for the elevations of US cities. Use a data smoothing
# technique since the elevations are for points, and "spread" those values across areas.**

# %%
import geopandas as gpd
import geoplot as gplt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np

# Load the usa_cities dataset
path = gplt.datasets.get_path("usa_cities")
usa_cities = gpd.read_file(path)

# Filter for continental US cities
continental_usa_cities = usa_cities.query('STATE not in ["HI", "AK", "PR"]')

# Extract city coordinates
points = np.array([(point.x, point.y) for point in continental_usa_cities.geometry])
elevations = continental_usa_cities["ELEV_IN_FT"].to_numpy()

# Generate valid Voronoi polygons and associate with elevations
vor = Voronoi(points)
vor_polygons = []
valid_elevations = []

for region_idx, elevation in zip(vor.point_region, elevations):
    region = vor.regions[region_idx]
    if -1 not in region and region:  # Exclude infinite or empty regions
        polygon = Polygon([vor.vertices[i] for i in region])
        if polygon.is_valid:  # Ensure the polygon is valid
            vor_polygons.append(polygon)
            valid_elevations.append(elevation)

# Create GeoDataFrame with valid polygons and corresponding elevations
polygons_with_elevation = gpd.GeoDataFrame(
    {"geometry": vor_polygons, "Elevation": valid_elevations},
    crs=continental_usa_cities.crs
)

# Load US states shapefile and compute boundary
us_states = gpd.read_file(gplt.datasets.get_path("contiguous_usa"))
us_boundary = us_states.geometry.unary_union

# Clip Voronoi polygons to the US border
clipped = gpd.overlay(
    polygons_with_elevation,
    gpd.GeoDataFrame(geometry=[us_boundary], crs=us_states.crs),
    how="intersection"
)

# Plot the Voronoi diagram colored by elevation
fig, ax = plt.subplots(figsize=(12, 8))
clipped.plot(column="Elevation", cmap="terrain", ax=ax, legend=True, legend_kwds={'label': "Elevation (ft)"})

# Plot US state boundaries
us_states.boundary.plot(ax=ax, linewidth=1, edgecolor="black", label="State Boundaries")

# Plot cities as points
continental_usa_cities.plot(ax=ax, color="red", markersize=5, label="Cities")

# Add title, legend, and axis labels
plt.title("Voronoi Diagram for the elevation above sea level in the US, based on cities' elevation data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(loc="upper right", frameon=True, fontsize=10)

# Show grid for better reference of latitude and longitude
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.show()


# %% [markdown]
# The Voronoi diagram displays the region's elevation around cities in the US. Based on the data about each city the Voronoi technique suggests elevation level about larger areas/regions. We have tried to make the visualization self explainatory by describing axis, legend, keeping cities and state boarders in accurate position to eachother etc.
# 
# As the instructuion was to create a Voronoi diagram for the elevations of US cities the Voronoi diagram might not be the optimal choice to showcase indivudual cities specific elevation. It is hard to see the exact elevation of a city for three reasons. First, a city may be located on the border between two colors. Second, the legend used is based on a continuous color scale rather than based on color associated binning. Third, the Voronoi reveales additional information regarding the area between the cities. Example of an mprovement would be to make the map "interactable" so that hoovering over a city data point would give the exact elevation or use anpther visualization technique that points out seperate cities elevation. However, this kind of visualization is very informative when it comes to getting a good overwiew of the elevation of certain specific geograhic points and the elevation's variation. 
# 
# When adding on smoothing geometric buffer we noticed that the size of the colored regions expanded, but not the cities or state boarders. When testing different levels of smoothening, we observed that a too high smoothing level could cause a color overlap that mislead the interpretation of the elevation in the surrounding areas. We therfore, decided to only have a low degree of smooting. 

# %% [markdown]
# **Q5. Apply the same Covid19 visualization section’s method for visualizing the
# COVID-19 recovered case dataset**

# %%
# Load the dataset with recovered global COVID-19 cases
df_recoveredGlobal = pd.read_csv("..\\..\\3.4Covid-19\\time_series_covid19_recovered_global.csv")
print(df_recoveredGlobal.head())

# Remove unnecessary columns and sum by country
df_recoveredGlobal = df_recoveredGlobal.drop(columns=['Province/State', 'Lat', 'Long'])
df_recoveredGlobal = df_recoveredGlobal.groupby('Country/Region').agg('sum')
date_list_recovered = list(df_recoveredGlobal.columns)

# Add country codes to the dataset
df_recoveredGlobal['country'] = df_recoveredGlobal.index
df_recoveredGlobal['iso_alpha_3'] = df_recoveredGlobal['country'].apply(get_country_code)

# Transform to long format
df_long_recovered = pd.melt(df_recoveredGlobal, 
                            id_vars=['country', 'iso_alpha_3'], 
                            value_vars=date_list_recovered)
print(df_long_recovered)

# Create Map with Plotly Express
fig_recovered = px.choropleth(df_long_recovered,
                              locations="iso_alpha_3",
                              color="value",
                              hover_name="country",
                              animation_frame="variable",
                              projection="natural earth",
                              color_continuous_scale='Greens',
                              range_color=[0, 50000],
                              labels={"value": "Covid-19 Recovered cases", "variable": "Date"}
                              )

fig_recovered.show()
fig_recovered.write_html("Covid19_recovered_map.html") # write it to HTML file

# %% [markdown]
# This map illstrates how the total number of recovered Covid-19 patients evolved around the world over time. It does not take the number of deceased into account, but only the number of recovered in thousands. The graphics concearn the time span from 01/22/2020 to 11/06/2020.

# %% [markdown]
# **Q6. Apply the same Covid19 visualization section’s method for visualizing the
# COVID-19 death case dataset**

# %%
# Load the dataset with global COVID-19 death cases
df_deathsGlobal = pd.read_csv("..\\..\\3.4Covid-19\\time_series_covid19_deaths_global.csv")
print(df_deathsGlobal.head())

# Remove unnecessary columns and sum by country
df_deathsGlobal = df_deathsGlobal.drop(columns=['Province/State', 'Lat', 'Long'])
df_deathsGlobal = df_deathsGlobal.groupby('Country/Region').agg('sum')
date_list_deaths = list(df_deathsGlobal.columns)

# Add country codes to the dataset
df_deathsGlobal['country'] = df_deathsGlobal.index
df_deathsGlobal['iso_alpha_3'] = df_deathsGlobal['country'].apply(get_country_code)

# Transform to long format
df_long_deaths = pd.melt(df_deathsGlobal, 
                         id_vars=['country', 'iso_alpha_3'], 
                         value_vars=date_list_deaths)
print(df_long_deaths)

# Create Map with Plotly Express
fig_deaths = px.choropleth(df_long_deaths,
                           locations="iso_alpha_3",
                           color="value",
                           hover_name="country",
                           animation_frame="variable",
                           projection="natural earth",
                           color_continuous_scale='Reds',
                           range_color=[0, 50000],
                           labels={"value": "Covid-19 Death cases", "variable": "Date"}
                           )

fig_deaths.show()
fig_deaths.write_html("C:/Users/emmae/Desktop/DataVisualization_Uppgiftsmapp/Labb2/D7055E-Labb-2/notebooks/Covid19_deaths_map.html") # write it to HTML file


# %% [markdown]
# This map illstrates how the total number of deth cases due to Covid-19 evolved around the world over time. It does not take the number of deceased into account, but only the number of dead in thousands. The graphics concearn the time span from 01/22/2020 to 11/06/2020.


