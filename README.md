# LI_Project

## Overview
This project involves the analysis and prediction of bike path lengths in Amsterdam and Krakow using spatial data and machine learning. The aim is to create a model that can predict the bike path lengths in Krakow based on the features extracted from Amsterdam's data.

## Project Structure
- main.py: The main script to run the entire analysis and prediction workflow.
- data_processing.py: Contains functions to preprocess spatial data, including generating H3 hex grids, cropping grids, calculating bike path lengths, and adding additional features.
- model_training.py: Contains functions for training and evaluating the machine learning model.
- prediction.py: Contains functions for applying the trained model to predict bike path lengths in Krakow.
- plots.py: Contains functions for visualizing the data and results.
- requirements.txt: Lists the required Python packages for the project.


## Instalation

### Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install the required packages:
pip install -r requirements.txt

### Data Preparation
Ensure you have the required parquet files (amsterdam_bike_paths_extended.parquet and krakow_bike_paths_extended.parquet) in the project directory.

### Usage
#### Run the main.py script:
python main.py
