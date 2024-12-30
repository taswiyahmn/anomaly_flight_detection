import streamlit as st
import pandas as pd
from preprocessing import DataPreprocessor, display_csv_data
from kmeans import KMeansClusteringApp
from birch import BirchClusteringApp
from gmm import GMMClusteringApp
from abnormality import AbnormalityVisualizationApp

class App:
    def __init__(self):
        self.data = None
        self.preprocessing_done = False
        self.preprocessed_data = None

    def preprocess_data_for_both(self, data):
        if data is not None:
            preprocessor = DataPreprocessor(data)
            columns_to_drop = ["icao24", "squawk", "radar", "time", "departure", "destination", "on_ground", "airline_icao"]
            preprocessor.drop_column_data(columns_to_drop)
            preprocessor.filter_altitude()
            preprocessor.convert_data()
            preprocessor.update_heading()
            preprocessor.calculate_haversine_distance()
            preprocessor.calculate_elevation()
            preprocessor.calculate_vertical_speed_difference()
            preprocessor.add_label_column()
            preprocessor.distance_fix()
            preprocessor.drop_na_distance()
            preprocessor.calculate_avg_elevation()
            preprocessor.calculate_avg_altitude()
            preprocessor.calculate_avg_vs()
            preprocessor.tidy_up_label()

            columns_to_drop_after = ["flight_id", "latitude", "longitude", "altitude", "number", "airline_iata", 
                                     "latitude_rad", "longitude_rad", "altitude_meter", "haversine", "elevation", 
                                     "distance", "vertical_speed", "diff_vs", "heading"]
            preprocessor.drop_column_data(columns_to_drop_after)

            preprocessor.label_avg_elevation()
            preprocessor.label_vertical_speed()

            return preprocessor.df
        else:
            return None

    def run(self):
        st.sidebar.title("Navigation")
        menu = st.sidebar.selectbox("Select a feature", ["Menu", "Preprocessing and Modeling Process", "Abnormality Visualization"])

        if menu == "Menu":
            st.title("Welcome to Flight Analysis Data")
            st.write("Use the sidebar to navigate to different features.")
            
        elif menu == "Preprocessing and Modeling Process":
            self.upload_and_preprocess()
            self.clustering()

        elif menu == "Abnormality Visualization":
            self.upload_and_preprocess()
            self.visualize_abnormality()

    def upload_and_preprocess(self):
        st.sidebar.title("Upload CSV File")
        uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

        st.title("Flight Data Preprocessing and Clustering")

        self.data = display_csv_data(uploaded_file)

        if st.button("Run Preprocessing"):
            self.preprocessed_data = self.preprocess_data_for_both(self.data)

            if self.preprocessed_data is not None:
                self.preprocessing_done = True
                st.session_state['preprocessing_done'] = True
                st.session_state['preprocessed_data'] = self.preprocessed_data

                st.write("### Data after Preprocessing:")
                st.write(self.preprocessed_data)
            else:
                st.write("No data to preprocess.")

    def clustering(self):
        if st.session_state.get('preprocessing_done', False):
            st.sidebar.title("Choose Clustering Method")
            clustering_method = st.sidebar.selectbox("Clustering Method", ["Select a method", "KMeans", "GMM", "BIRCH"])

            if clustering_method != "Select a method":
                if clustering_method == "KMeans":
                    kmeans_model = KMeansClusteringApp(st.session_state['preprocessed_data'])
                    kmeans_model.run_clustering()
                elif clustering_method == "GMM":
                    gmm_model = GMMClusteringApp(st.session_state['preprocessed_data'])
                    gmm_model.run_clustering()
                elif clustering_method == "BIRCH":
                    birch_model = BirchClusteringApp(st.session_state['preprocessed_data'])
                    birch_model.run_clustering()
            else:
                st.write("Please select a clustering method from the dropdown.")
        else:
            st.write("Please run the preprocessing step first.")

    def visualize_abnormality(self):
        if st.session_state.get('preprocessing_done', False):
            st.write("### Visualizations based on Abnormality:")
            
            abnormality_visualizer = AbnormalityVisualizationApp(st.session_state['preprocessed_data'])
            
            st.write("#### Elevation Abnormality Visualization:")
            abnormality_visualizer.visualize_abnormality_elevation()

            st.write("#### Vertical Speed Abnormality Visualization:")
            abnormality_visualizer.visualize_abnormality_vertical_speed()
        else:
            st.write("No data to visualize abnormality. Please preprocess the data.")

if __name__ == "__main__":
    app = App()
    app.run()