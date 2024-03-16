import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import osmnx as ox
from shapely.geometry import Point
from shapely.geometry import LineString
import shapely.wkt
from utilities import *
from tqdm import tqdm
import time

start = time.time()
def simplify_graph_geometry(G):
    for u, v, data in G.edges(data=True):
        if 'geometry' in data:
            # Convert the WKT string to a shapely LineString object
            line = shapely.wkt.loads(data['geometry'])

            if isinstance(line, LineString):
                # Simplify the LineString to just the start and end points
                data['geometry'] = LineString([line.coords[0], line.coords[-1]])

    return G

fraction = 0.6
date_list = ["2015-05-04", "2015-05-05", "2015-05-06", "2015-05-07", "2015-05-08", "2015-05-11", "2015-05-12", "2015-05-13", "2015-05-14", "2015-05-15"]
dict_date = {}
column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                        'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                        'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']

# Fetching the boundary of Manhattan
city = ox.geocode_to_gdf('Manhattan, New York, USA')
city_polygon = city['geometry'].iloc[0]

def is_within_manhattan(lng, lat):
    point = Point(lng, lat)  # Note the order: (longitude, latitude)
    return city_polygon.contains(point)

def process_row(row):
    origin_grid_id = assign_neighborhood_ids(row['origin_lng'], row['origin_lat'], manhattan_nta, neighborhood_to_id)
    dest_grid_id = assign_neighborhood_ids(row['dest_lng'], row['dest_lat'], manhattan_nta, neighborhood_to_id)
    origin = row['origin_id']
    dest = row['dest_id']
    itinerary_node_list = []
    itinerary_segment_dis_list = []

    # Check if the itinerary exists in the database
    query = {
        'origin': origin, 
        'destination': dest
    }
    re = mycollection.find_one(query)
    if re:
        ite = re['itinerary_node_list']
    else:
        # Compute shortest path using OSMnx
        ite = ox.distance.shortest_path(G_manhattan, origin, dest, weight='length', cpus=32)
        if not ite:
            ite = [origin, dest]
        content = {
            'origin': origin,
            'destination': dest,
            'itinerary_node_list': ite
        }
        try:
            mycollection.insert_one(content)
        except Exception as e:
            print(f"Error inserting data for origin: {origin}, destination: {dest}: {e}")

    if ite is not None and len(ite) > 1:
        itinerary_node_list.append(ite)
    else:
        itinerary_node_list.append([origin, dest])

    # Calculate segment distances
    itinerary_segment_dis = []
    for i in range(len(ite) - 1):
        dis = distance(node_id_to_coord[ite[i]], node_id_to_coord[ite[i + 1]])
        itinerary_segment_dis.append(dis)
    itinerary_segment_dis_list.append(itinerary_segment_dis)

    # Return the computed values to be assigned back to the DataFrame
    return origin_grid_id, dest_grid_id, itinerary_node_list, itinerary_segment_dis_list, sum(itinerary_segment_dis)

for date in tqdm(date_list):
    with open(f'./input_generation/1NYU_{date}.csv', 'rb') as f:
        df = pd.read_csv(f)
    df = df[column_name]
    sampled_df = df.sample(frac=fraction, random_state=42)
    print(f"{date} data sampled")
    # Filter rows based on geolocation
    sampled_df = sampled_df[sampled_df.apply(lambda row: is_within_manhattan(row['origin_lng'], row['origin_lat']) and 
                     is_within_manhattan(row['dest_lng'], row['dest_lat']), axis=1)]
    print(f"{date} data filtered")

    # Apply the process_row function to each row in the DataFrame
    results = sampled_df.apply(lambda row: process_row(row), axis=1)
    # Unpack results into separate DataFrame columns
    sampled_df['origin_grid_id'], sampled_df['dest_grid_id'], sampled_df['itinerary_node_list'], sampled_df['itinerary_segment_dis_list'], sampled_df['trip_distance'] = zip(*results)

     # Initialize the dictionary with keys from 0 to 86400 and empty lists as values
    dict_time = {i: [] for i in range(86401)}
    # Filter out rows where 'start_time' is NaN
    sampled_df = sampled_df[sampled_df['start_time'].notna()]
    # Convert each row in the DataFrame to a list and append to the corresponding 'start_time' key in the dictionary
    sampled_df.apply(lambda row: dict_time[row['start_time']].append(row.tolist()), axis=1)
    # Assign the dict_time to corresponding date
    dict_date[date] = dict_time
    print(f"{date} successfully created")

# Serialize the sampled orders
with open(f'input_generation/orders_{fraction}_complete.pickle', 'wb') as f:
    pickle.dump(dict_date, f)

end = time.time()
print(f"Execution time: {end - start} seconds")
