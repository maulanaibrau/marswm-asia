from streamlit_folium import folium_static, st_folium
import streamlit as st
import streamlit.components.v1 as components
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


st.markdown("#### Rice Yield and Maximum Flood Depth Map", unsafe_allow_html=True)

# load map
rootFolder = os.path.join(os.getcwd(), "01_apps", "subang")
riceMapPath = os.path.join(rootFolder, "riceyield-prediction", "Subang.shp")
riceMapGpd = gpd.read_file(riceMapPath)
gdf = riceMapGpd.set_geometry("geometry")
riceMapGpd = gdf.to_crs("EPSG:4326")

colors = ['#ffa500', '#f0e68c', '#ffff00', '#90ee00', '#00ff00', '#006400']
thresholds = [500, 550, 600, 650, 700, 750]
colormap = cm.LinearColormap(colors = colors, 
                                vmin = min(thresholds), 
                                vmax = max(thresholds), 
                                index = thresholds,
                                caption = "Rice Yield Prediction (kg/10a)")
step_colormap = colormap.to_step(index = thresholds, method = 'quantiles')

riceMapGpdExplore = riceMapGpd.explore(
    column = 'Harvest',
    tooltip = ['id_new', 'Harvest'],
    cmap = step_colormap,
    name = "Rice Yield Prediction (kg/10a)"
)

waterDepthPath = os.path.join(rootFolder, "numericalsimulation-max-waterdepth", "Cimacan_Cell.shp")
waterDepthMap = gpd.read_file(waterDepthPath)
gdf = waterDepthMap.set_geometry("geometry")
waterDepthMap = gdf.to_crs("EPSG:4326")

waterdepth_colors = [mcolors.to_rgba('white', 0.1), '#cce0ff', '#a0c4ff', '#3c8cff', '#66a3ff', '#0073e6', '#004080']
waterdepth_thresholds = [0, 0.01, 0.2, 0.4, 0.6, 1, 3]
waterdepth_colormap = cm.LinearColormap(colors = waterdepth_colors, 
                            vmin = min(waterdepth_thresholds), 
                            vmax = max(waterdepth_thresholds), 
                            index = waterdepth_thresholds,
                            caption = "180 mm Max. Flood Depth (m)")
step_waterdepth_colormap = waterdepth_colormap.to_step(index = waterdepth_thresholds, method = 'quantiles')

waterDepthMap.explore(
    m = riceMapGpdExplore,
    column = '180mm',
    tooltip = ['CN', '90mm', '180mm'],
    # cmap = step_colormap,
    color = "blue",
    name = "180 mm Max Flood Depth (m)"
)

folium.TileLayer("CartoDB positron", show=False).add_to(riceMapGpdExplore)
folium.LayerControl().add_to(riceMapGpdExplore)
folium_static(riceMapGpdExplore)