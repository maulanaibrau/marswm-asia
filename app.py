import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import tensorflow as tf
import matplotlib.pyplot as plt
import folium
import os
import glob
import tempfile

from keras.models import Sequential     
from keras.layers import Dense
from keras.layers import LSTM
from streamlit_folium import st_folium

from helper import readData

st.image("eAsia_logo.png")
st.title("eAsia Joint Research Project 2021-2024")
#st.title("eAsia Project Machine Learning Streamlit Web App")
#st.subheader("LSTM Method for Inland Flood Modelling")
st.markdown("Development of Machine Learning and Remote Sensing-based Water Management Platform in Asian Deltas for Sustainable Agriculture")
st.markdown("Website: [MARSWM-ASIA](https://marswm-asia.net/)")
st.markdown("---")

st.subheader("Machine Learning Program (LSTM Method) for Inland Flood Prediction")
st.text("Train your data before making predictions.")
st.markdown("Start by visiting this link: [Train](https://colab.research.google.com/drive/185Uy862ntZjZCJp3VvIba0sJ7xSoWpb1#scrollTo=DpRCNBjIcIIc)")

with st.sidebar.form("my_form"):
    training_file = st.file_uploader("Choose the training file (.h5)")
    validation_file = st.text_input("Choose a folder containing rainfall files for testing (.csv)")
    calibration_file = st.text_input("Choose a folder containing water depth files for calibration (.csv)")

    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if validation_file is not None:
            all_validation_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(validation_file) for f in filenames]
            length_validation_files = len(all_validation_files)
            

        if calibration_file is not None:
            all_calibration_files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(calibration_file) for f in filenames]
            length_calibration_file = len(all_calibration_files)
            
            train = []
            for i in range(length_validation_files):
                train.append(readData(all_validation_files[i], all_calibration_files[i]))

            train_rains = [train[i][0] for i in range(length_validation_files)]
            train_cells = [train[i][1] for i in range(length_calibration_file)]
            np_rains = np.array(train_rains)

            dir_norm = np.array([1.388841111111111, 
                                4.7358875642616765, 
                                -1.1852149822042217,
                                1.3252189963416747,
                                46.409108796296294,
                                43.61695646840649])
            df_norm = pd.DataFrame(dir_norm)

            rain_mean = np.squeeze(df_norm.values).tolist()[0]
            rain_std = np.squeeze(df_norm.values).tolist()[1]

            # z-score
            train_rains_zscore = (np_rains - rain_mean) / rain_std

            X_train = train_rains_zscore
            Y_train = np.array(train_cells)
    # else:
    #     st.text("Please upload all data from the sidebar")

        # omission of warning
        if training_file is not None:
            temp_model_path = os.path.join(tempfile.gettempdir(), "temp_model.h5")
            with open(temp_model_path, "wb") as temp_model_file:
                temp_model_file.write(training_file.read())

            mdl = Sequential()
            mdl.add(LSTM(X_train.shape[1], activation='tanh', input_shape=(X_train.shape[1], 1)))
            mdl.add(Dense(X_train.shape[1]))
            mdl.compile(optimizer='adam', loss='mse')
            prediction = mdl.predict(X_train, batch_size=10)


tab1, tab2, tab3 = st.tabs(['Your Own Prediction :books:','Kameda, Japan :jp:','Xuan Thuy, Vietnam 🇻🇳'])

with tab1:

    if submitted:


        fig, ax = plt.subplots(1, figsize=(15, 8))
        ax_2 = ax.twinx()

        # Slider to select the file index
        selected_file_index = st.slider("Select File Index", 0, X_train.shape[0] - 1, 0)

        
        # # Plotting the selected file
        # ax.plot(X_train[selected_file_index, :, 0], label=f'File {selected_file_index + 1}')

        # # Load validation data from files in the specified folder
        # validation_data, _ = readData(validation_file)

        # Plotting the selected file
        ax_2.plot(np.arange(X_train.shape[1]), color='orange', label='Validation Data')



        # ax.set_xlabel('Time Step')
        # ax.set_ylabel('Rainfall')
        # ax_2.set_ylabel('Validation Data')

        # ax.legend(loc='upper left')
        # ax_2.legend(loc='upper right')

        st.pyplot(fig)

        st.text("Model Summary")
        mdl.summary(print_fn=lambda x: st.text(x))
        st.code(prediction, language="python", line_numbers=True)
    else:
        st.text("Please upload all data from the sidebar")

with tab2:
    st.image("Kameda Watershed.png")

    rootFolder = os.path.join(os.getcwd(), "01_apps")
    subFolders = [name for name in os.listdir(rootFolder) if os.path.isdir(os.path.join(rootFolder, name))]

    rainFolderName = st.selectbox(
        "Choose the rain folder",
        subFolders,
        #index = None,
        label_visibility = "hidden",
        placeholder = "Choose the rain folder"
    )

    selectedRainDirectory = os.path.join(rootFolder, rainFolderName, "WL-Cell")
    listOfCellWaterLevelFiles = [os.path.splitext(filename)[0] for filename in os.listdir(selectedRainDirectory)]
    cellWaterLevelFile = st.selectbox(
        "Choose the water level file",
        listOfCellWaterLevelFiles,
        #index = None,
        label_visibility = "hidden",
        placeholder = "Choose the water level file"
    )

    # load csv
    selectedFile = os.path.join(rootFolder, rainFolderName, "WL-Cell", cellWaterLevelFile + '.csv')
    waterLevelValues = pd.read_csv(selectedFile)
    waterLevelValues["step"] = "step" + waterLevelValues["step"].apply(str)
    waterLevelValues = waterLevelValues.T
    waterLevelValues.columns = waterLevelValues.iloc[0]
    waterLevelValues = waterLevelValues[1:]
    waterLevelValuesCN = waterLevelValues.assign(CN=range(1, len(waterLevelValues)+1, 1))

    # load map
    kamedaMapPath = os.path.join(rootFolder, "Kameda_Cells.shp")
    kamedaMap = gpd.read_file(kamedaMapPath)
    gdf = kamedaMap.set_geometry("geometry")
    kamedaMap = gdf.to_crs("EPSG:4326")

    # join csv + map on CN
    kamedaMerge = gpd.GeoDataFrame(waterLevelValuesCN.merge(kamedaMap, on="CN"))
    # Centering the map automatically (based on our features)
    x_map = kamedaMerge.centroid.x.mean()
    y_map = kamedaMerge.centroid.y.mean()

    selectedStep = st.select_slider(
        'Simulate the value of water level', 
        options = waterLevelValues.columns
    )

    # dissolve kameda by its step
    kamedaMergeDissolve = kamedaMerge[[selectedStep, 'CN', 'geometry', 'Area']]
    kamedaMergeDissolve = kamedaMerge.dissolve(
        by=selectedStep,
        aggfunc={
            "CN": "count",
            "Area": "sum"
        }
    )
    kamedaMergeDissolve = kamedaMergeDissolve.reset_index()
    
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
        fill_color="YlGnBu",                # cmap
        name="Kameda",
        show=True,
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    st_folium(m)

    # kamedaMergeDissolve["geometry"] = kamedaMergeDissolve.geometry.simplify(0.05)
    # kamedaExplore = kamedaMergeDissolve.explore(
    #     column = f'{selectedStep}', 
    #     scheme = "naturalbreaks",
    #     legend = True,
    #     k = 5,
    #     tooltip = f"{selectedStep}",
    #     popup = [f"{selectedStep}", "Area"],
    #     legend_kwds = dict(colorbar=False), 
    #     name = "Kameda"
    # )
    # folium.TileLayer("CartoDB positron", show=False).add_to(kamedaExplore)
    # folium.LayerControl().add_to(kamedaExplore)
    # st_folium(kamedaExplore)

    fig, ax = plt.subplots(1,1)
    kamedaMerge.plot(
        column=selectedStep,
        # legend=True,
        ax=ax,
        cmap='rainbow'
    )
    ax.set_title("LTSM " + selectedStep)
    st.pyplot(fig)
    
    code = cellWaterLevelFile.split("_")[0]
    rainFallPath = glob.glob(os.path.join(rootFolder, rainFolderName, "Rainfall", code + "*"))[0]
    rainFallData = pd.read_csv(rainFallPath, names=["rainfall"])
    WvMlPath = glob.glob(os.path.join(rootFolder, rainFolderName, "WL-ML", code + "*"))[0]
    WvMlData = pd.read_csv(WvMlPath, names=["wv_ml"])
    WvPmPath = glob.glob(os.path.join(rootFolder, rainFolderName, "WL-PM", code + "*"))[0]
    WvPmPData = pd.read_csv(WvPmPath, names=["wv_pm"])

    figs, axs = plt.subplots(3)
    axs[0].plot(rainFallData)
    axs[0].set_title("Rainfall")
    axs[1].plot(WvMlData)
    axs[1].set_title("Water Volume from Machine Learning Model")
    axs[2].plot(WvPmPData)
    axs[2].set_title("Water Volume from Physical Model")

    st.pyplot(figs)

    