import streamlit as st
import numpy as np
import pandas as pd

def display_csv_data(file):
    if file is not None:
        df = pd.read_csv(file)
        st.write("### Data Preview:")
        st.write(df)
        return df
    else:
        st.write("Upload a CSV file to see its contents.")

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def drop_column_data(self, columns_to_drop):
        if self.df is not None:
            self.df = self.df.drop(columns_to_drop, axis=1)
        else:
            st.write("No data to preprocess.")

    def filter_altitude(self):
        if self.df is not None:
            self.df = self.df[self.df['altitude'] != 0]
        else:
            st.write("No data to filter.")

    def convert_data(self):
        if self.df is not None:
            self.df['latitude_rad'] = np.radians(self.df['latitude'])
            self.df['longitude_rad'] = np.radians(self.df['longitude'])
            self.df['altitude_meter'] = self.df['altitude'] * 0.3048
        else:
            st.write("No data to convert.")

    def update_heading(self):
        self.df['Runway'] = ''  # Initialize 'Runway' column with empty strings
        self.df.loc[(self.df['heading'] >= 90) & (self.df['heading'] <= 270), 'Runway'] = 18
        self.df.loc[((self.df['heading'] >= 0) & (self.df['heading'] <= 90)) | ((self.df['heading'] >= 270) & (self.df['heading'] <= 360)), 'Runway'] = 36

    def calculate_haversine_distance(self):
        if self.df is not None:
            R = 6371000  # Radius of the Earth
            self.df['lat2'] = np.where(self.df['Runway'] == 18, 0.008154, np.where(self.df['Runway'] == 36, 0.007877, np.nan))
            self.df['lon2'] = np.where(self.df['Runway'] == 18, 1.770544, np.where(self.df['Runway'] == 36, 1.770536, np.nan))

            dlat = self.df['lat2'] - self.df['latitude_rad']
            dlon = self.df['lon2'] - self.df['longitude_rad']
            a = np.sin(dlat / 2)**2 + np.cos(self.df['latitude_rad']) * np.cos(self.df['lat2']) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c

            self.df['haversine'] = distance
            self.df.drop(['lat2', 'lon2'], axis=1, inplace=True)
        else:
            st.write("No data to calculate Haversine distance.")

    def calculate_elevation(self):
        if self.df is not None:
            self.df['elevation'] = np.degrees(np.arctan(self.df['altitude_meter'] / self.df['haversine']))
        else:
            st.write("No data to calculate Elevation.")

    def calculate_vertical_speed_difference(self):
        if self.df is not None:
            self.df['diff_vs'] = self.df['vertical_speed'].abs().diff().abs()
            self.df['diff_vs'].fillna(0, inplace=True)
        else:
            st.write("No data provided.")

    def add_label_column(self):
        self.df['label_pesawat'] = (self.df['registration'] != self.df['registration'].shift(1)) | (self.df['callsign'] != self.df['callsign'].shift(1))
        self.df['label_pesawat'] = self.df['label_pesawat'].cumsum()

    def distance_fix(self):
        distances = []

        for label, group in self.df.groupby('label_pesawat'):
            haversine_values = group['haversine'].tolist()

            for i in range(len(haversine_values) - 1):
                if haversine_values[i] > haversine_values[i + 1]:
                    distances.append(haversine_values[i])
                else:
                    distances.append(haversine_values[i])
                    break  # Stop the loop after finding the minimum value

            for _ in range(i + 1, len(haversine_values)):
                distances.append(np.nan)

        # Create a new 'distance' column in the original DataFrame
        self.df['distance'] = distances

    def drop_na_distance(self):
        self.df.dropna(subset=['distance'], inplace=True)

    def tidy_up_label(self):
        self.df = self.df.groupby('label_pesawat').first().reset_index()

    def calculate_avg_elevation(self):
        avg_elevation_df = self.df.groupby('label_pesawat')['elevation'].mean().reset_index()
        avg_elevation_df['avg_elevation'] = avg_elevation_df['elevation']  # Create a new column
        self.df = self.df.merge(avg_elevation_df[['label_pesawat', 'avg_elevation']], on='label_pesawat')

    def calculate_avg_altitude(self):
        avg_altitudes_m = self.df.groupby('label_pesawat')['altitude_meter'].mean().reset_index()
        avg_altitudes_m['avg_altitude_m'] = avg_altitudes_m['altitude_meter']  # Create a new column
        self.df = self.df.merge(avg_altitudes_m[['label_pesawat', 'avg_altitude_m']], on='label_pesawat')

    def calculate_avg_vs(self):
        avg_diffs_vs = self.df.groupby('label_pesawat')['diff_vs'].mean().reset_index()
        avg_diffs_vs['avg_diff_vs'] = avg_diffs_vs['diff_vs']  # Create a new column
        self.df = self.df.merge(avg_diffs_vs[['label_pesawat', 'avg_diff_vs']], on='label_pesawat')

    def label_avg_elevation(self):
        if self.df is not None:
            self.df['label_avg_elevation'] = 'abnormal'  # Default to 'abnormal'
            self.df.loc[(self.df['avg_elevation'] >= 2.5) & (self.df['avg_elevation'] <= 3.5), 'label_avg_elevation'] = 'normal'
            return self.df
        else:
            st.write("No data to label average elevation.")

    def label_vertical_speed(self):
        if self.df is not None:
            self.df['label_vertical_speed'] = 'abnormal'  # Default to 'abnormal'
            self.df.loc[(self.df['avg_diff_vs'] >= 60) & (self.df['avg_diff_vs'] <= 180), 'label_vertical_speed'] = 'normal'
            return self.df
        else:
            st.write("No data to label vertical speed.")