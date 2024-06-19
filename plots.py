import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd

def plot_h3_grid(gdf_hex, city_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    city_area = ox.geocode_to_gdf(city_name)
    city_area.plot(ax=ax, facecolor="none", edgecolor="black")
    gdf_hex.plot(ax=ax, alpha=0.5, edgecolor="k")
    plt.title(f"H3 Grid for {city_name}")
    plt.show()

def plot_feature_distribution(gdf, column_name):
    gdf.plot(column=column_name, legend=True, cmap='viridis', figsize=(10, 10))
    plt.title(f'Distribution of {column_name}')
    plt.show()

def plot_comparison_map(gdf_original, gdf_predicted, column_original, column_predicted):
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    gdf_original.plot(column=column_original, ax=axes[0], legend=True, cmap='viridis', edgecolor='k')
    axes[0].set_title('Original Bike Path Lengths')

    gdf_predicted.plot(column=column_predicted, ax=axes[1], legend=True, cmap='viridis', edgecolor='k')
    axes[1].set_title('Predicted Bike Path Lengths')

    plt.show()