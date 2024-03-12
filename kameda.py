from streamlit_folium import folium_static, st_folium
import streamlit as st
# import streamlit.components.v1 as components
from st_pages import add_page_title, hide_pages

import pandas as pd
import numpy as np
import geopandas as gpd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import branca.colormap as cm

add_page_title() #layout="wide"

hide_pages(["Thank you"])

css = '''
<style>
    
</style>
'''

#st.image("Kameda Watershed.png")
st.markdown(css, unsafe_allow_html=True)

tab1, tab2 = st.tabs(['Map','Model'])

with tab1:
    st.markdown("#### Rice Yield and Maximum Flood Depth Map", unsafe_allow_html=True)

    # load map
    rootFolder = os.path.join(os.getcwd(), "01_apps", "kameda")
    riceMapPath = os.path.join(rootFolder, "riceyield-prediction", "paddy", "Kameda2023_Paddy.shp")
    riceMapGpd = gpd.read_file(riceMapPath)
    gdf = riceMapGpd.set_geometry("geometry")
    riceMapGpd = gdf.to_crs("EPSG:4326")

    colors = ['#ffa500', '#f0e68c', '#ffff00', '#90ee00', '#00ff00', '#006400']
    thresholds = [500, 550, 600, 650, 700, 750]
    colormap = cm.LinearColormap(colors = colors, 
                                 vmin = min(thresholds), 
                                 vmax = max(thresholds), 
                                 index = thresholds,
                                 caption = "Predicted Yield, 2023")
    step_colormap = colormap.to_step(index = thresholds, method = 'quantiles')

    riceMapGpdExplore = riceMapGpd.explore(
        column = 'Harvest',
        tooltip = ['id', 'Harvest'],
        cmap = step_colormap,
        name = "Rice Yield Prediction (kg/10a)"
    )

    waterDepthPath = os.path.join(rootFolder, "numericalsimulation-max-waterdepth", "Kameda_Cells.shp")
    waterDepthMap = gpd.read_file(waterDepthPath)
    gdf = waterDepthMap.set_geometry("geometry")
    waterDepthMap = gdf.to_crs("EPSG:4326")

    waterDepthMap.explore(
        m = riceMapGpdExplore,
        column = '300mm',
        tooltip = ['CN', '100mm', '200mm', '300mm'],
        # cmap = step_colormap,
        color = "blue",
        name = "300 mm Max Flood Depth (m)"
    )

    folium.TileLayer("CartoDB positron", show=False).add_to(riceMapGpdExplore)
    folium.LayerControl().add_to(riceMapGpdExplore)
    folium_static(riceMapGpdExplore)

with tab2:

    st.markdown("#### LSTM-based Inland Flood Model", unsafe_allow_html=True)

    rainFolder = os.path.join(rootFolder, "cellwl_lstm")
    subFolders = [name for name in os.listdir(rainFolder) if os.path.isdir(os.path.join(rainFolder, name))]
    rainFolderName = st.selectbox(
        "Choose the Rainfall (mm) Event for Simulation",
        subFolders,
        #index = None,
        #label_visibility = "hidden",
        placeholder = "Choose the rain folder"
    )

    selectedRainDirectory = os.path.join(rainFolder, rainFolderName)
    listOfCellWaterLevelFiles = [os.path.splitext(filename)[0] for filename in os.listdir(selectedRainDirectory)]
    cellWaterLevelFile = st.selectbox(
        "Choose the Water Depth (m) Scenario",
        listOfCellWaterLevelFiles,
        #index = None,
        #label_visibility = "hidden",
        placeholder = "Choose the water level file"
    )

    # load csv
    selectedFile = os.path.join(rainFolder, rainFolderName, cellWaterLevelFile + '.csv')
    waterLevelValues = pd.read_csv(selectedFile)
    waterLevelValues["step"] = "step" + waterLevelValues["step"].apply(str)
    waterLevelValues = waterLevelValues.T
    waterLevelValues.columns = waterLevelValues.iloc[0]
    waterLevelValues = waterLevelValues[1:]
    waterLevelValuesCN = waterLevelValues.assign(CN=range(1, len(waterLevelValues)+1, 1))

    # load map
    kamedaMapPath = os.path.join(rootFolder, "numericalsimulation-max-waterdepth\Kameda_Cells.shp")
    kamedaMap = gpd.read_file(kamedaMapPath)
    gdf = kamedaMap.set_geometry("geometry")
    kamedaMap = gdf.to_crs("EPSG:4326")

    # join csv + map on CN
    kamedaMerge = gpd.GeoDataFrame(waterLevelValuesCN.merge(kamedaMap, on="CN"))
    # Centering the map automatically (based on our features)
    x_map = kamedaMerge.centroid.x.mean()
    y_map = kamedaMerge.centroid.y.mean()

    selectedStep = st.select_slider(
        'Choose the Time Step (seconds) for Water Level Simulation with 30 minutes (1,800 seconds) interval. Maximum simulation duration is 72 hours (257,400 seconds or 3 days).', 
        options = waterLevelValues.columns
    )

    # dissolve kameda by its step
    kamedaMergeDissolve = kamedaMerge[[selectedStep, 'CN', 'geometry', rainFolderName]]
    kamedaMergeDissolve = kamedaMerge.dissolve(
        by=selectedStep,
        aggfunc={
            "CN": "count",
            rainFolderName: "sum"
        }
    )
    kamedaMergeDissolve = kamedaMergeDissolve.reset_index()

    colors = ['#ffffff', '#cce0ff', '#66a3ff', '#0073e6', '#004080']
    thresholds = [0, 0.01, 0.2, 0.4, 0.6, 1]

    # Define LinearColormap with specified colors and thresholds
    colormap = cm.LinearColormap(colors=colors, vmin=min(thresholds), vmax=max(thresholds))

    # Convert LinearColormap to StepColormap
    step_colormap = colormap.to_step(index=thresholds, method='quantiles')
    # -------------
    # Convert StepColormap to a string representing a color
    #fill_color = colormap.to_linear()

    # Convert StepColormap to a list of colors
    #fill_color = [colormap(x) for x in np.linspace(0, 1, 6)]  # Adjust the number of colors as needed
    fill_color = step_colormap.to_linear()
    # -------------
    # Convert LinearColormap to a string representing a colormap name
    cmap_name = fill_color

    # initiate the basemap
    m = folium.Map(location=[y_map, x_map],  # center of the folium map
                zoom_start=12,            # initial zoom
                tiles="CartoDB positron") # type of map
    # folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(m)
    # folium.GeoJson("kameda4.geojson").add_to(m)

    folium.Choropleth(
        geo_data=kamedaMergeDissolve,       # geo data
        data=kamedaMergeDissolve,           # data
        columns=["CN", selectedStep],       # [key, value]
        key_on="feature.properties.CN",     # feature.properties.key
        #fill_color="YlGnBu",               # cmap
        #fill_color=colormap,               # use custom color scale
        #fill_color=step_colormap,          # Use the converted color scale
        cmap=colormap,
        #fill_color = colormap(0),          # Select the first color from the list
        fill_opacity=0.7,                   # adjust opacity
        line_opacity=0.2,                   # adjust line opacity
        name="Kameda",
        show=True,
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)

    # st_folium(m)


    #Geopanda/Matplotlib based Map -----------
    # Define custom color ranges and corresponding colors
    bounds = [0, 0.01, 0.2, 0.4, 0.6, 1]
    colors = ['#FFFFFF', '#ADD8E6', '#87CEEB', '#0000FF', '#00008B']

    # Create a colormap
    cmap = mcolors.ListedColormap(colors)

    # Assuming kamedaMerge['selectedStep'] contains the data for flood depth
    data_min = kamedaMerge[selectedStep].min()
    data_max = kamedaMerge[selectedStep].max()

    # Ensure the data range falls within the bounds
    if data_min < bounds[0]:
        bounds[0] = data_min
    if data_max > bounds[-1]:
        bounds[-1] = data_max

    # Create a norm object to define the range of your data
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Create a dummy scatter plot to get a mappable object
    dummy = plt.scatter([], [], c=[], cmap=cmap, norm=norm)

    # Plot the Matplotlib/Geopandas data
    fig1, ax = plt.subplots(1,1)
    kamedaMerge.plot(
        column=selectedStep,
        legend=False,
        #legend_kwds={"Flood Inundation Depth (m)"},
        ax=ax,
        cmap=cmap,
        norm=norm,
        # cmap='rainbow'
        # cmap=cmap_name
    )

    # Add colorbar
    cbar = plt.colorbar(dummy,ax=ax, orientation='vertical')
    # cbar.set_label('Labellll')

    ax.set_title("Flood Inundation Depth: " + selectedStep)
    # Adjust the font size of x and y ticks
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

    # Add titles to x and y axes with units
    plt.xlabel("Longitude (°)", fontsize=9)
    plt.ylabel("Latitude (°)", fontsize=9)

    # st.pyplot(fig1)

    code = cellWaterLevelFile.split("_")[3]
    rainFallPath = glob.glob(os.path.join(rootFolder, "rainfall", rainFolderName, "*" + code + "*"))[0]
    rainFallData = pd.read_csv(rainFallPath, names=["rainfall"])
    WvMlPath = glob.glob(os.path.join(rootFolder, "wv_lstm", rainFolderName, "*" + code + "*"))[0]
    WvMlData = pd.read_csv(WvMlPath, names=["wv_ml"])
    WvPmPath = glob.glob(os.path.join(rootFolder, "wv_numericalsimulation", rainFolderName, "*" + code + "*"))[0]
    WvPmPData = pd.read_csv(WvPmPath, names=["wv_pm"])

    t = np.arange(0, len(WvPmPData), 1)

    # Calculate RMSE and NSE
    rmse = np.sqrt(np.mean((WvPmPData.values - WvMlData.values)**2))
    mean_observed = np.mean(WvPmPData.values)
    nse = 1 - np.sum((WvMlData.values - WvPmPData.values)**2) / np.sum((WvPmPData.values - mean_observed)**2)
    # Convert RMSE to units of 10^6 m^3
    rmse_million_m3 = rmse * 1e-6

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)

    ax1.plot(t, WvMlData, "-b", label='Machine learning') 
    ax1.plot(t, WvPmPData, "-r", label='Numerical simulation') 

    ax1.grid(True)
    ax1.set_xlabel('Time (30 minutes)')
    ax1.set_ylabel('Flood inundation volume')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    rainFallData.insert(1, "time", t)
    #ax2.hist(rainFallData, edgecolor="black", label="rainfall")
    ax2.bar(rainFallData['time'], rainFallData['rainfall'], label="Rainfall", alpha=0.6, edgecolor='grey')
    ax2.invert_yaxis()
    ax2.set_ylabel('Rainfall')  

    # Include RMSE and NSE in the plot
    plt.text(0.5, 0.05, f'RMSE: {rmse_million_m3:.2f} x 10^6 m3', transform=ax1.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')
    plt.text(0.5, 0.1, f'NSE: {nse:.2f}', transform=ax1.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')

    #fig.legend()
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=0)

    # st.pyplot(fig2)

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        st.pyplot(fig1)
    with col2:
        st.pyplot(fig2)