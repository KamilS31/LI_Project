import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import osmnx as ox
import geopandas as gpd
import pandas as pd
import os

from data_processing import create_h3_hex_grid, crop_hex_grid, calculate_bike_path_lengths, add_additional_features
from model_training import train_models, evaluate_model
from prediction import apply_model_to_krakow
from plots import plot_h3_grid, plot_comparison_map
from sklearn.model_selection import train_test_split, GridSearchCV
from shapely.ops import unary_union

def preprocess_data(amsterdam_bounds, krakow_bounds):
    print("1) Generate H3 grids for Amsterdam and Krakow")
    amsterdam_grid = create_h3_hex_grid(amsterdam_bounds)
    krakow_grid = create_h3_hex_grid(krakow_bounds)

    # Plot H3 grids before cropping
    print("Plotting H3 grids for Amsterdam and Krakow before cropping")
    plot_h3_grid(amsterdam_grid, 'Amsterdam, Netherlands')
    plot_h3_grid(krakow_grid, 'Kraków, Poland')

    print("2) Crop H3 grids to city boundaries")
    amsterdam_grid = crop_hex_grid(amsterdam_grid, 'Amsterdam, Netherlands', 3857)
    krakow_grid = crop_hex_grid(krakow_grid, 'Kraków, Poland', 3857)

    # Plot H3 grids after cropping
    print("Plotting H3 grids for Amsterdam and Krakow after cropping")
    plot_h3_grid(amsterdam_grid, 'Amsterdam, Netherlands')
    plot_h3_grid(krakow_grid, 'Kraków, Poland')

    print("3) Calculate bike path lengths for each hex")
    amsterdam_data = calculate_bike_path_lengths(amsterdam_grid, 'amsterdam_bike_paths_extended.parquet')
    krakow_data = calculate_bike_path_lengths(krakow_grid, 'krakow_bike_paths_extended.parquet')

    krk_numh3 = krakow_data['h3_index'].nunique()
    krk_bike_sum = krakow_data['bike_path_length'].sum()
    print(f"Number of unique H3 indexes: {krk_numh3}")
    print(f"Sum of bike path lengths: {krk_bike_sum}")

    ams_numh3 = amsterdam_data['h3_index'].nunique()
    ams_bike_sum = amsterdam_data['bike_path_length'].sum()
    print(f"Number of unique H3 indexes: {ams_numh3}")
    print(f"Sum of bike path lengths: {ams_bike_sum}")

    print("4) Add additional features")
    hex_area_amsterdam = unary_union(amsterdam_grid.geometry)
    hex_area_krakow = unary_union(krakow_grid.geometry)
    amsterdam_data = add_additional_features(amsterdam_bounds, amsterdam_data, hex_area_amsterdam, city='Amsterdam')
    krakow_data = add_additional_features(krakow_bounds, krakow_data, hex_area_krakow, city='Krakow')

    print(krakow_data)

    # Save preprocessed data
    amsterdam_data.to_csv('Amsterdam_data.csv', index=False)
    krakow_data.to_csv('Krakow_data.csv', index=False)

    return amsterdam_data, krakow_data

def main():
    amsterdam_bounds = {"north": 52.441157, "south": 52.2688, "east": 5.1127658, "west": 4.728073}
    krakow_bounds = {"north": 50.1257, "south": 49.9639, "east": 20.215, "west": 19.7946}

    with mlflow.start_run():
        amsterdam_path = 'Amsterdam_data.csv'
        krakow_path = 'Krakow_data.csv'

        # Preprocess data only if the preprocessed files do not exist
        if not os.path.exists(amsterdam_path) or not os.path.exists(krakow_path):
            amsterdam_data, krakow_data = preprocess_data(amsterdam_bounds, krakow_bounds)
        else:
            print("Loading preprocessed data")
            amsterdam_data = pd.read_csv(amsterdam_path)
            krakow_data = pd.read_csv(krakow_path)

        print(krakow_data.head(5))
        print("5) Train models")
        # Create training and validation sets
        X = amsterdam_data.drop(columns=['bike_path_length', 'h3_index', 'geometry'])
        y = amsterdam_data['bike_path_length']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        amsterdam_model = train_models(X_train, y_train)

        # Generate predictions on the validation set
        predictions = amsterdam_model.predict(X_val)

        # Evaluate the model
        evaluate_model(predictions, y_val)

        print("6) Apply model to Krakow data")
        X_krakow = krakow_data.drop(columns=['bike_path_length', 'h3_index', 'geometry'])
        predictions_krakow = apply_model_to_krakow(amsterdam_model, X_krakow)

        # Add predictions to Krakow data
        krakow_data['predicted_bike_path_length'] = predictions_krakow

        # Convert krakow_data to GeoDataFrame
        krakow_data_gdf = gpd.GeoDataFrame(krakow_data, geometry=gpd.GeoSeries.from_wkt(krakow_data['geometry']))
        krakow_data_gdf.set_crs(epsg=4326, inplace=True)

        print("7) Plot results")
        plot_comparison_map(krakow_data_gdf, krakow_data_gdf, 'bike_path_length', 'predicted_bike_path_length')


if __name__ == "__main__":
    main()
