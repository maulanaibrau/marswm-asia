import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tempfile
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import matplotlib.pyplot as plt

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

    fig, ax1 = plt.subplots(1, figsize=(15, 8))
    ax_21 = ax1.twinx()
    #


    # # Assuming X_train is your normalized rainfall data for all files
    # num_files = X_train.shape[0]

    # # Slider to select the file index
    # selected_file_index = st.slider("Select File Index", 0, num_files - 1, 0)

    # # Flattening the 3D array to a 2D array for the selected file
    # selected_file_rainfall = X_train[selected_file_index, :, :].reshape(X_train.shape[1], -1)

    # # Assuming you have corresponding calibration data for each file
    # selected_file_calibration = Y_train[selected_file_index, :]

    # # Creating DataFrames
    # chart_rainfall_data = pd.DataFrame(selected_file_rainfall, columns=['Rainfall at Time Step {}'.format(i) for i in range(selected_file_rainfall.shape[1])])
    # chart_calibration_data = pd.DataFrame({'Water Depth': selected_file_calibration})

    # # Plotting the line charts
    # st.line_chart(chart_rainfall_data, height=650, use_container_width=True)

    # # Adding a separate axis for the calibration data
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # ax2.plot(chart_calibration_data.index, chart_calibration_data['Water Depth'], 'r-')
    # ax2.set_ylabel('Water Depth', color='red')
    # st.pyplot(fig)


# plot data
#chart_data = pd.DataFrame(np.random.randn(10,3),columns=['a','b','c'])
#st.line_chart(chart_data, height=650)
