import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

class AbnormalityVisualizationApp:
    def __init__(self, df):
        self.df = df

    def count_abnormalities(self, label_column):
        abnormal_count = self.df[self.df[label_column] == 'abnormal'].shape[0]
        normal_count = self.df[self.df[label_column] == 'normal'].shape[0]
        return abnormal_count, normal_count

    def visualize_abnormality_vertical_speed(self):
        fig, ax = plt.subplots(figsize=(20, 5))

        # Color mapping dictionary
        color_map = {'abnormal': 'red', 'normal': 'blue'}

        # Plot data points with color based on 'label_vertical_speed'
        scatter = ax.scatter(
            x=self.df['avg_elevation'], 
            y=self.df['avg_diff_vs'], 
            c=self.df['label_vertical_speed'].map(color_map), 
            edgecolor='black', 
            alpha=0.7
        )

        # Plot settings
        ax.set_title('Elevation vs Vertical Speed - Abnormality based on Elevation')
        ax.set_xlabel('Average Difference Vertical Speed')
        ax.set_ylabel('Average Elevation ')
        ax.legend(handles=scatter.legend_elements()[0], labels=['normal', 'abnormal'])

        # Display the plot
        ax.grid(True)
        st.pyplot(fig)

        # Calculate total counts of abnormalities and normals
        abnormal_count = (self.df['label_vertical_speed'] == 'abnormal').sum()
        normal_count = (self.df['label_vertical_speed'] == 'normal').sum()
        st.write("### vertical speed Abnormality Count")
        st.table(pd.DataFrame({
            'Type': ['Vertical speed Abnormality'],
            'Abnormal': [abnormal_count],
            'Normal': [normal_count]
        }))

        # Count occurrences based on 'aircraft_code' and 'label_vertical_speed'
        count_df = self.df.groupby(['aircraft_code', 'label_vertical_speed']).size().unstack(fill_value=0).reset_index()
        count_df.columns = ['aircraft_code', 'Abnormal', 'Normal']

        # Display the counts for the top 10 aircraft codes
        st.write("### Aircraft Code Counts with vertical speed Abnormalities and Normals")
        st.table(count_df.head(10))

    def visualize_abnormality_elevation(self):
    
        fig, ax = plt.subplots(figsize=(20, 5))

        # Color mapping dictionary
        color_map = {'abnormal': 'red', 'normal': 'blue'}

        # Plot data points with color based on 'label_avg_elevation'
        scatter = ax.scatter(
            x=self.df['avg_elevation'], 
            y=self.df['avg_diff_vs'], 
            c=self.df['label_avg_elevation'].map(color_map), 
            edgecolor='black', 
            alpha=0.7
        )

        # Plot settings
        ax.set_title('Elevation vs Vertical Speed - Abnormality based on Vertical Speed')
        ax.set_xlabel('Average Elevation')
        ax.set_ylabel('Average Difference Vertical Speed')
        ax.legend(handles=scatter.legend_elements()[0], labels=['abnormal', 'normal'])

        # Display the plot
        ax.grid(True)
        st.pyplot(fig)

        # Calculate total counts of abnormalities and normals
        abnormal_count = (self.df['label_avg_elevation'] == 'abnormal').sum()
        normal_count = (self.df['label_avg_elevation'] == 'normal').sum()
        st.write("### Elevation Abnormality Count")
        st.table(pd.DataFrame({
            'Type': ['Elevation Abnormality'],
            'Abnormal': [abnormal_count],
            'Normal': [normal_count]
        }))

        # Count occurrences based on 'aircraft_code' and 'label_avg_elevation'
        count_df = self.df.groupby(['aircraft_code', 'label_avg_elevation']).size().unstack(fill_value=0).reset_index()
        count_df.columns = ['aircraft_code', 'Abormal', 'Normal']

        # Display the counts for the top 10 aircraft codes
        st.write("### Aircraft Code Counts with Abnormalities and Normals")
        st.table(count_df.head(10))


# Usage
# df = pd.read_csv('your_data.csv')
# app = AbnormalityVisualizationApp(df)
# app.visualize_abnormality_elevation()
# app.visualize_abnormality_vertical_speed()
