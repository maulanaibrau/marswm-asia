import streamlit as st
import pandas as pd
import numpy as np
import os

def read_data(training_path, validation_path):
    df_rain0 = pd.read_csv(training_path, header=None)
    df_rain = df_rain0
    lst_rain = df_rain.values.tolist()
    lst_rain_t = np.array(lst_rain).reshape(-1, 1).tolist()
    np_rain = np.array(lst_rain_t)
    
    df_cell0 = pd.read_csv(validation_path, header=None)
    df_cell = df_cell0
    lst_cell = df_cell.values.tolist()
    np_cell = np.array(lst_cell)

    return np_rain, np_cell

def readData(training_path, validation_path):
    df_rain0 = pd.read_csv(training_path, header=None)
    np_rain = df_rain0.iloc[:,0].values
    np_rain = np_rain[:, np.newaxis]
    
    df_cell0 = pd.read_csv(validation_path, index_col=0)
    df_cell = df_cell0
    lst_cell = df_cell.values.tolist()
    np_cell = np.array(lst_cell)
    
    return np_rain, np_cell

def file_selector(folder_path = '.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.sidebar.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def save_df_to_folder(df, folder_path, file_name):
    """Saves dataframe to the provided folder."""
    if not os.path.isdir(folder_path):
        st.error('The provided folder does not exist. Please provide a valid folder path.')
        return

    file_path = os.path.join(folder_path, file_name)
    df.to_csv(file_path, index=False)
    st.success(f'Successfully saved dataframe to {file_path}')