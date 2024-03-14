import streamlit as st
from st_pages import add_page_title, hide_pages
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

from helper import readData

add_page_title() #layout="wide"

hide_pages(["Thank you"])

css = '''
<style>
    [data-testid='stFileUploader'] {
        width: max-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

with st.form("my_form"):
    training_file = st.file_uploader("Choose the training files (.h5)", type="zip")

    col1, col2 = st.columns(2, gap='medium')
    with col1:
        validation_file = st.text_input("Choose a folder containing rainfall files for testing (.csv)")

    with col2:
        calibration_file = st.text_input("Choose a folder containing water depth files for calibration (.csv)")

    cell_area_file = st.file_uploader("Choose the cell area files (.csv)", type="csv")

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

        if cell_area_file is not None:
            dfa = pd.read_csv(cell_area_file, header=None, index_col=0)
            lst_dfa = np.squeeze(dfa.values).tolist()
            np_dfa = np.array([lst_dfa])
            np_dfa_rp = np.repeat(np_dfa, 144, axis=0)


        # omission of warning
        if training_file is not None:
            myzipfile = zipfile.ZipFile(training_file)
            with tempfile.TemporaryDirectory() as tmp_dir:
                myzipfile.extractall(tmp_dir)
                root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
                model_dir = os.path.join(tmp_dir, root_folder)
                #st.info(f'trying to load model from tmp dir {model_dir}...')
                model = tf.keras.models.load_model(model_dir)

            lst_nse_all = []
            lst_nse_ave_all = []

            # for i0 in range(len(training_file)):
            #prediction = model.predict(X_train[0:28])
            prediction = model.predict(X_train, batch_size=1)
            lst_nse = []
            for i in range(length_validation_files):

                sim_volume = np.multiply(prediction[i],np_dfa_rp)
                sim_volume_sum = np.sum(sim_volume, axis=1) # SIMULATION WATER VOLUME

                ob_volume = np.multiply(Y_train[i],np_dfa_rp)
                ob_volume_sum = np.sum(ob_volume, axis=1) # CALLIBRATION

                n = len(ob_volume_sum)
                mean_observed = np.mean(ob_volume_sum)
                lst_mean_observed = np.repeat(mean_observed, 144, axis=0) # THE FINAL RESULT

                lst_nse_u = []
                lst_nse_l = []

                for t in range(n):
                    nse_u = (ob_volume_sum[t] - sim_volume_sum[t])**2
                    nse_l = (ob_volume_sum[t] - lst_mean_observed[t])**2
                    lst_nse_u.append(nse_u)
                    lst_nse_l.append(nse_l)

                #
                sum_lst_nse_u = sum(lst_nse_u)
                sum_lst_nse_l = sum(lst_nse_l)
                
                nse = 1-(sum_lst_nse_u/sum_lst_nse_l)

                print("data:", i, "nse:", nse)

                # if nse < 0.7:
                #     plot(prediction[i],np_dfa_rp,Y_train[i],np_rains[i])
                
                lst_nse.append(nse)

            lst_nse_all = lst_nse_all + lst_nse
            lst_nse_ave_all.append(sum(lst_nse)/length_validation_files)

            df_nse_all = pd.concat([pd.Series(list(range(len(lst_nse_all)))), pd.Series(lst_nse_all)], axis=1)
            df_nse_all.columns = ['data', 'NSE']
            # st.table(df_nse_all)
            st.dataframe(df_nse_all.set_index(df_nse_all.columns[0]), use_container_width=True)

            listOfData = st.selectbox(
                "Choose dataset",
                list(range(length_validation_files)),
                #index = None,
                #label_visibility = "hidden",
                placeholder = "Choose dataset"
            )

            c1, c2 = st.columns(2, gap='medium')
            with c1:
                x = np.max(prediction[listOfData], axis=0)
                y = np.max(Y_train[listOfData], axis=0)

                fig, ax = plt.subplots()
                
                plt.plot(x, y, 'o')
                m, b = np.polyfit(x, y, 1)
                plt.plot(x, m*x+b)

                # plt.text(.02, .8, r'$y$='+str(m)+r'$x$'+str(b))
                ax.annotate(r'$R^2={:.3f}$'.format(r2_score(y, x)), (.4, .6))
                ax.set_xlabel("Numerical Simulation Modelling-based Max. Inundation Depth (m)", fontsize=7)
                ax.set_ylabel("LSTM Modelling-based Max. Inundation Depth (m)", fontsize=7)
                ax.set_title("Rainfall scenario")

                st.pyplot(fig)
            with c2:
                rainFallData = pd.DataFrame(train_rains[0], columns=['rainfall'])
                WvPmPData = pd.DataFrame(ob_volume_sum, columns=['wv_pm'])
                WvMlData = pd.DataFrame(sim_volume_sum, columns=["wv_ml"])

                t = np.arange(0, len(WvPmPData), 1)

                # Calculate RMSE and NSE
                rmse = np.sqrt(np.mean((WvPmPData.values - WvMlData.values)**2))
                mean_observed = np.mean(WvPmPData.values)
                nse = 1 - np.sum((WvMlData.values - WvPmPData.values)**2) / np.sum((WvPmPData.values - mean_observed)**2)
                # Convert RMSE to units of 10^6 m^3
                rmse_million_m3 = rmse * 1e-6

                fig = plt.figure()
                ax1 = fig.add_subplot(111)

                ax1.plot(t, WvMlData, "-b", label='Machine learning') # , 'machine learning'
                ax1.plot(t, WvPmPData, "-r", label='Numerical simulation') # , 'physical model'

                ax1.grid(True)
                ax1.set_xlabel('Time (30 minutes)')
                ax1.set_ylabel('Flood inundation volume')

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
                rainFallData.insert(1, "time", t)
                #ax2.hist(rainFallData, edgecolor="black", label="rainfall")
                ax2.bar(rainFallData['time'], rainFallData['rainfall'], label="Rainfall", alpha=0.6, edgecolor='grey')
                #ax2.legend()
                ax2.invert_yaxis()
                ax2.set_ylabel('Rainfall')  

                plt.text(0.5, 0.05, f'RMSE: {rmse_million_m3:.2f} x 10^6 m3', transform=ax1.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')
                plt.text(0.5, 0.1, f'NSE: {nse:.2f}', transform=ax1.transAxes, fontsize=8, verticalalignment='center', horizontalalignment='center')

                #fig.legend()
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1+h2, l1+l2, loc=0)

                st.pyplot(fig)

