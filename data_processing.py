import geopandas as gpd
import pandas as pd
from h3 import h3
import osmnx as ox
from shapely.geometry import Polygon, LineString, MultiLineString, mapping, Point
from shapely.ops import transform, unary_union
from pyproj import Transformer
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from plots import plot_feature_distribution


def create_h3_hex_grid(bounds, epsg=4326, resolution=7):
    h3_indices = h3.polyfill(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    [bounds["west"], bounds["north"]],
                    [bounds["east"], bounds["north"]],
                    [bounds["east"], bounds["south"]],
                    [bounds["west"], bounds["south"]],
                    [bounds["west"], bounds["north"]],
                ]
            ],
        },
        resolution,
    )

    hexagons = []
    for h in h3_indices:
        hex_boundary = h3.h3_to_geo_boundary(h, geo_json=True)
        hex_boundary = [(lng, lat) for lat, lng in hex_boundary]
        hex_boundary.append(hex_boundary[0])
        hexagons.append({
            'geometry': Polygon(hex_boundary),
            'h3_index': h
        })

    gdf_hex = gpd.GeoDataFrame(hexagons)
    gdf_hex = gdf_hex.set_crs(epsg=epsg)

    return gdf_hex

def crop_hex_grid(gdf_hex, city_name, epsg):
    city_area = ox.geocode_to_gdf(city_name)
    city_area = city_area.to_crs(epsg=epsg)

    gdf_hex = gdf_hex.to_crs(epsg=epsg)
    gdf_hex['within_city'] = gdf_hex['geometry'].apply(lambda x: city_area.geometry.intersects(x).any())

    cropped_gdf_hex = gdf_hex[gdf_hex['within_city']].copy()
    cropped_gdf_hex = cropped_gdf_hex.to_crs(epsg=4326)
    cropped_gdf_hex.drop(columns=['within_city'], inplace=True)
    cropped_gdf_hex.set_geometry('geometry', inplace=True)

    return cropped_gdf_hex

def calculate_bike_path_lengths(grid, parquet_file):
    bike_paths = gpd.read_parquet(parquet_file)
    bike_paths = bike_paths.to_crs(epsg=3857)
    grid = grid.to_crs(epsg=3857)

    grid['bike_path_length'] = 0.0

    for i, polygon in grid.iterrows():
        clipped = bike_paths.clip(polygon.geometry)
        grid.at[i, 'bike_path_length'] = clipped.length.sum()

    grid = grid.to_crs(epsg=4326)
    return grid[['h3_index', 'bike_path_length', 'geometry']]

def get_city_center(city_name):
    geolocator = Nominatim(user_agent="city_center_locator")
    location = geolocator.geocode(city_name)
    center_gdf = gpd.GeoDataFrame(geometry=[Point(location.longitude, location.latitude)], crs=4326)
    if location:
        return center_gdf
    else:
        return None

def calculate_distance_to_center(hex, city):
    center_gdf = get_city_center(city)
    center_gdf = center_gdf.to_crs(epsg=3857)
    center = center_gdf.geometry.centroid.iloc[0]

    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    hex_center = transformer.transform(*h3.h3_to_geo(hex))

    return ox.distance.euclidean_dist_vec(center.y, center.x, hex_center[1], hex_center[0])

def calculate_road_lengths(city_bounds, gdf_hex, hex_area):
    city_roads = ox.graph_from_bbox(
        bbox=(city_bounds['north'], city_bounds['south'], city_bounds['east'], city_bounds['west']),
        network_type='drive')
    city_walks = ox.graph_from_bbox(
        bbox=(city_bounds['north'], city_bounds['south'], city_bounds['east'], city_bounds['west']),
        network_type='walk')

    roads = ox.graph_to_gdfs(city_roads, nodes=False)
    walks = ox.graph_to_gdfs(city_walks, nodes=False)

    roads['highway'] = roads['highway'].apply(lambda x: x[0] if isinstance(x, list) else x)
    main_road_types = ['secondary', 'primary', 'tertiary', 'busway', 'motorway_link', 'motorway']
    main_roads = roads[roads['highway'].isin(main_road_types)]

    main_roads_clipped = gpd.clip(main_roads, hex_area)
    walks_clipped = gpd.clip(walks, hex_area)

    gdf_hex = gdf_hex.to_crs(epsg=28992)
    main_roads_clipped = main_roads_clipped.to_crs(epsg=28992)
    walks_clipped = walks_clipped.to_crs(epsg=28992)

    gdf_hex['main_roads_length'] = 0.0
    gdf_hex['walks_length'] = 0.0

    for i, polygon in gdf_hex.iterrows():
        clipped_r = main_roads_clipped.clip(polygon.geometry)
        clipped_w = walks_clipped.clip(polygon.geometry)
        gdf_hex.at[i, 'main_roads_length'] = clipped_r.length.sum()
        gdf_hex.at[i, 'walks_length'] = clipped_w.length.sum()

    gdf_hex = gdf_hex.to_crs(epsg=4326)
    return gdf_hex

def calculate_green_space_areas(city_bounds, gdf_hex, hex_area):
    green_spaces = ox.features_from_bbox(north=city_bounds['north'], south=city_bounds['south'],
                                           east=city_bounds['east'], west=city_bounds['west'],
                                           tags={'leisure': 'park', 'landuse': ['recreation_ground', 'forest'],
                                                 'natural': 'wood'})

    green_spaces_clipped = gpd.clip(green_spaces, hex_area)

    gdf_hex = gdf_hex.to_crs(epsg=28992)
    green_spaces_clipped = green_spaces_clipped.to_crs(epsg=28992)

    gdf_hex['green_space_area'] = 0.0

    for i, polygon in gdf_hex.iterrows():
        clipped_g = gpd.clip(green_spaces_clipped, polygon.geometry)
        gdf_hex.at[i, 'green_space_area'] = clipped_g.area.sum()

    gdf_hex = gdf_hex.to_crs(epsg=4326)
    return gdf_hex

def calculate_service_amenities(city_bounds, gdf_hex, hex_area):
    service_amenity_tags = {'amenity': True, 'shop': True, 'office': True}

    service_amenities = ox.features_from_bbox(north=city_bounds['north'], south=city_bounds['south'],
                                                east=city_bounds['east'], west=city_bounds['west'],
                                                tags=service_amenity_tags)

    service_amenities_clipped = gpd.clip(service_amenities, hex_area)

    gdf_hex['service_amenity_count'] = 0

    for i, polygon in gdf_hex.iterrows():
        clipped_amenities = gpd.clip(service_amenities_clipped, polygon.geometry)
        gdf_hex.at[i, 'service_amenity_count'] = clipped_amenities.shape[0]

    return gdf_hex

def calculate_population_density(city_bounds, gdf_hex, hex_area):
    population_data = ox.features_from_bbox(north=city_bounds['north'], south=city_bounds['south'],
                                              east=city_bounds['east'], west=city_bounds['west'],
                                              tags={'population': True})

    population_data_clipped = gpd.clip(population_data, hex_area)

    gdf_hex['population_density'] = 0

    for i, polygon in gdf_hex.iterrows():
        clipped_population = gpd.clip(population_data_clipped, polygon.geometry)
        clipped_population['population'] = pd.to_numeric(clipped_population['population'], errors='coerce').fillna(0)
        gdf_hex.at[i, 'population_density'] = clipped_population['population'].sum() / polygon.geometry.area

    return gdf_hex

def plot_feature_distribution(gdf_hex, feature_name):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_hex.plot(column=feature_name, ax=ax, legend=True, cmap='viridis')
    plt.title(f"Distribution of {feature_name}")
    plt.show()

def add_additional_features(city_bounds, gdf_hex, hex_area, city):
    gdf_hex = gpd.GeoDataFrame(gdf_hex)  # Ensure gdf_hex is a GeoDataFrame
    gdf_hex.set_geometry("geometry", inplace=True)  # Set the geometry column
    gdf_hex.set_crs(epsg=4326, inplace=True)  # Set CRS if not already set

    print("Calculating road lengths...")
    gdf_hex = calculate_road_lengths(city_bounds, gdf_hex, hex_area)
    gdf_hex.to_csv(f'{city}_data.csv', index=False)
    #plot_feature_distribution(gdf_hex, 'main_roads_length')

    print("Calculating green space areas...")
    gdf_hex = calculate_green_space_areas(city_bounds, gdf_hex, hex_area)
    gdf_hex.to_csv(f'{city}_data.csv', index=False)
    #plot_feature_distribution(gdf_hex, 'green_space_area')

    print("Calculating service amenities...")
    gdf_hex = calculate_service_amenities(city_bounds, gdf_hex, hex_area)
    gdf_hex.to_csv(f'{city}_data.csv', index=False)
    #plot_feature_distribution(gdf_hex, 'service_amenity_count')

    print("Calculating population density...")
    gdf_hex = calculate_population_density(city_bounds, gdf_hex, hex_area)
    gdf_hex.to_csv(f'{city}_data.csv', index=False)
    #plot_feature_distribution(gdf_hex, 'population_density')

    print("Calculating distance to city center...")
    gdf_hex['distance_to_city_center'] = gdf_hex['h3_index'].apply(lambda x: calculate_distance_to_center(x, city))
    gdf_hex.to_csv(f'{city}_data.csv', index=False)
    #plot_feature_distribution(gdf_hex, 'distance_to_city_center')

    return gdf_hex
