import pandas as pd
import numpy as np
import pickle
import random
from copy import deepcopy
from path import *
from utilities import *
from find_closest_point import *
from shapely.geometry import Point, Polygon
import sys

'''
    This util script was under 'simulator', now it is in 'simulator/test'. You may need
    to update path related codes in order to successfully run the code without errors.
'''

#Driver file
df_driver_info = pd.DataFrame(columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat','node_id' 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_node_id','target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list'])

# Get the shape of Manhattan
city = ox.geocode_to_gdf('Manhattan, New York, USA')
# project both GeoDataFrames to the same CRS (coordinate reference system)
gdf_nodes = gdf_nodes.to_crs(city.crs)
city = city.to_crs(gdf_nodes.crs)
# Create a boolean mask indicating which nodes are within the Manhattan polygon
mask = gdf_nodes['geometry'].apply(lambda x: city.geometry.contains(x).any())

# Define the coordinates of the bounding box (latitude, longitude) - replace with actual values
north, south, east, west = 40.8820, 40.8159, -73.9104, -73.9493
# Create a Polygon representing the bounding box
bbox_polygon = Polygon([(west, south), (west, north), (east, north), (east, south)])
# Create a boolean mask indicating which nodes are within the Harlem bounding box
north_mask = gdf_nodes['geometry'].apply(lambda x: bbox_polygon.contains(x))
# Invert the mask to get the nodes outside the Harlem bounding box
non_north_mask = ~north_mask
# Combine the Manhattan and non-Harlem masks 
final_mask = mask & non_north_mask

# Filter the nodes using the final mask
gdf_nodes_manhattan = gdf_nodes[final_mask]
# Sample the drivers from the filtered nodes
gdf_nodes_manhattan = gdf_nodes_manhattan.sample(n=env_params['driver_num'] * 2, replace=True)
# gdf_nodes = gdf_nodes.sample(n=env_params['driver_num'] * 2,replace = True)

lng_list = gdf_nodes_manhattan['x'].tolist()
lat_list = gdf_nodes_manhattan['y'].tolist()
id_list = gdf_nodes_manhattan.index.tolist()
df_driver_info['lng'] = lng_list[:env_params['driver_num']]
df_driver_info['lat'] = lat_list[:env_params['driver_num']]
origin_id_list = id_list[:env_params['driver_num']]
df_driver_info['driver_id'] = [str(i) for i in range(env_params['driver_num'])]
df_driver_info['start_time'] = env_params['t_initial']
df_driver_info['end_time'] = env_params['t_end']
df_driver_info['node_id'] = origin_id_list
df_driver_info['grid_id'] = [assign_neighborhood_ids(lng, lat, manhattan_nta, neighborhood_to_id) for lat, lng in zip(df_driver_info['lat'], df_driver_info['lng'])] # change grid id 
df_driver_info['status'] = 0
df_driver_info['target_loc_lng'] = lng_list[env_params['driver_num']:]
df_driver_info['target_loc_lat'] = lat_list[env_params['driver_num']:]
target_id_list = id_list[env_params['driver_num']:]
df_driver_info['target_node_id'] = target_id_list
# Create a dictionary mapping from node ID to (lat, lng)
id_to_lat_lng = pd.DataFrame({'lat': result['lat'], 'lng': result['lng']}).set_index(result['node_id']).T.to_dict('list')
# Now, update the 'target_grid_id' column
df_driver_info['target_grid_id'] = [assign_neighborhood_ids(id_to_lat_lng[node_id][1], id_to_lat_lng[node_id][0], manhattan_nta, neighborhood_to_id) for node_id in df_driver_info['target_node_id']]
df_driver_info['remaining_time'] = 0
df_driver_info['matched_order_id'] = 'None'
df_driver_info['total_idle_time'] = 0
df_driver_info['time_to_last_cruising'] = 0
df_driver_info['current_road_node_index'] = 0
df_driver_info['remaining_time_for_current_node'] = 0
df_driver_info['itinerary_node_list'] = [[] for _ in range(len(df_driver_info))]
df_driver_info['itinerary_segment_dis_list'] = [[] for _ in range(len(df_driver_info))]
print(df_driver_info)
pickle.dump(df_driver_info, open('input_generation/driver_100' + '.pickle', 'wb'))