# from re import I
# from socket import if_indextoname
import numpy as np
from copy import deepcopy
import random
from random import choice
from dispatch_alg import LD
from math import radians, sin, atan2,cos,acos
from config import *
import math
import pickle
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import pandas as pd
import sys
from collections import Counter
import pymongo
from pymongo.errors import ConnectionFailure
import time
import scipy.stats as st
from scipy.stats import skewnorm
from collections import deque
import os
from path import *
"""
Connect to mongoDB to speed up access to road network information
"""
myclient = pymongo.MongoClient("mongodb://localhost:27017/") # Use port: 27017
try:
    # The ismaster command is cheap and does not require auth.
    myclient.admin.command('ismaster')
    print("MongoDB is connected!")
except ConnectionFailure:
    print("Server not available")
mydb = myclient["manhattan_island"]
mycollection = mydb['manhattan_38grids']

"""
Load the information of graph network from graphml file.
Filter out the area of interest (Manhattan)
"""
G = ox.load_graphml(os.path.join(data_path, "manhattan.graphml"))
gdf_nodes, _ = ox.graph_to_gdfs(G)

# filter out Manhattan area
manhattan_polygon = ox.geocode_to_gdf('Manhattan, New York, USA')
nodes_geo = gpd.GeoDataFrame(gdf_nodes, geometry=[Point(xy) for xy in zip(gdf_nodes['x'], gdf_nodes['y'])])
nodes_in_manhattan = gpd.sjoin(nodes_geo, manhattan_polygon, how="inner", predicate='intersects')

# Extract the node IDs that are within the Manhattan polygon
manhattan_node_ids = nodes_in_manhattan.index.tolist()
# Create a subgraph of G that only contains nodes within Manhattan
G_manhattan = G.subgraph(manhattan_node_ids)

# map id to coordinate; map coordinate to node_id
node_id_to_coord = pd.Series(nodes_in_manhattan[['x', 'y']].apply(tuple, axis=1), index=nodes_in_manhattan.index).to_dict()
node_coord_to_id = {value: key for key, value in node_id_to_coord.items()}

# Extract 'node_id', 'lat' and 'lng' values
lng_list = nodes_in_manhattan['x'].tolist()
lat_list = nodes_in_manhattan['y'].tolist()
node_list = nodes_in_manhattan.index

"""
A dataframe `result` is generated to map coordinates to grid_id
"""
# Read NYC Neighborhood Tabulation Area (NTA) data
nta = gpd.read_file(os.path.join(data_path, "nynta2020_23d/nynta2020.shp"))
# Convert the CRS of the NTA data to match the osmnx graph
nta = nta.to_crs("EPSG:4326")
# Filter Manhattan neighborhoods
manhattan_nta = nta[nta['BoroName'] == 'Manhattan']
# Create a mapping from neighborhood names to unique integers
unique_neighborhoods = manhattan_nta['NTAName'].unique()
neighborhood_to_id = {neighborhood: i for i, neighborhood in enumerate(unique_neighborhoods)}
# Create a reverse mapping from ID to neighborhood name
id_to_neighborhood = {v: k for k, v in neighborhood_to_id.items()}

def assign_neighborhood_ids_array(lng_series, lat_series, manhattan_nta, neighborhood_to_id):
    '''
    Assign neighborhood IDs to a series of longitude and latitude coordinates.

    Parameters:
    - lng_series: pandas.Series containing longitude values
    - lat_series: pandas.Series containing latitude values
    - manhattan_nta: GeoDataFrame representing Manhattan neighborhoods
    - neighborhood_to_id: Dictionary mapping neighborhood names to IDs

    Returns:
    - A pandas.Series containing the assigned neighborhood IDs
    '''
    # Check if lng and lat series are of same length
    if len(lng_series) != len(lat_series):
        raise ValueError("Longitude and latitude series must be of the same length.")

    # Create Point objects from the longitude and latitude pairs
    points = gpd.GeoSeries([Point(x, y) for x, y in zip(lng_series, lat_series)], crs="EPSG:4326")

    # Create a GeoDataFrame from the points GeoSeries
    points_gdf = gpd.GeoDataFrame(geometry=points)

    # Spatially join the points GeoDataFrame and the neighborhood GeoDataFrame
    joined_gdf = gpd.sjoin(points_gdf, manhattan_nta, how="left", predicate='within')

    # Map the neighborhood names to IDs using the neighborhood_to_id dictionary
    # If NTAName does not exist in neighborhood_to_id, it returns NaN
    joined_gdf['grid_id'] = joined_gdf['NTAName'].map(neighborhood_to_id)

    # Handle points outside any neighborhood by setting their ID to -1
    joined_gdf['grid_id'] = joined_gdf['grid_id'].fillna(-1).astype(int)

    # Return the series of grid_ids
    return joined_gdf['grid_id'].reset_index(drop=True)

def assign_neighborhood_ids(lng, lat, manhattan_nta=manhattan_nta, neighborhood_to_id=neighborhood_to_id):
    '''
    '''
    # Create a Point object from the (latitude, longitude) pair
    point = Point(lng, lat)

    # Create a GeoDataFrame with the Point object
    point_gdf = gpd.GeoDataFrame([{'geometry': point}], crs="EPSG:4326")

    # Spatially join the points GeoDataFrame and the neighborhood GeoDataFrame
    joined_gdf = gpd.sjoin(point_gdf, manhattan_nta, how="left", predicate='within')

    # Map the neighborhood names to IDs using the neighborhood_to_id dictionary
    # If NTAName does not exist in neighborhood_to_id, it returns NaN
    joined_gdf['grid_id'] = joined_gdf['NTAName'].map(neighborhood_to_id)

    # Handle points outside any neighborhood by setting their ID to -1
    joined_gdf['grid_id'] = joined_gdf['grid_id'].fillna(-1).astype(int)

    # Return the grid_id for the single point, which should be a scalar integer
    return joined_gdf['grid_id'].iloc[0]
    
result = pd.DataFrame()
nodelist = []
result['lat'] = lat_list
result['lng'] = lng_list
result['node_id'] = node_list
# Assign neighborhood IDs to all points at once
result['grid_id'] = assign_neighborhood_ids_array(result['lng'], result['lat'], manhattan_nta, neighborhood_to_id)
# Create a mapping of grid_id to the first (lng, lat) entry
grid_id_to_first_coords = result.groupby('grid_id').first()[['lng', 'lat']].to_dict('index')
# Convert nested dictionary to a simpler dictionary
grid_id_to_first_coords = {grid: (coords['lng'], coords['lat']) for grid, coords in grid_id_to_first_coords.items()}
with open(os.path.join(data_path, "adjacent_neighbour_dict.pickle"), "rb") as f:
    grid_adjacency = pickle.load(f)
NUM_OF_GRIDS = 38

"""
Generate the available directions for each grid
"""
# Initialize the DataFrame with the required columns
df_available_directions = pd.DataFrame(columns=['zone_id', 'direction_0', 'direction_1', 'direction_2', 'direction_3', 'direction_4'])
# Calculate the centroid for each NTA
manhattan_nta['centroid'] = manhattan_nta.geometry.centroid
nta_centroids = manhattan_nta.set_index('NTAName')['centroid'].to_dict()

# Iterate over each NTA to determine the available directions
zone_id = []
centroid_lng = []
centroid_lat = []
up = [1, 3, 0, 4, 11, 9, 7, 8, 14, 10, 18, 12, 22, 15, 15, 16, 21, 16, 19, 20, 24, 23, 29, 29, 25, 26, 32, 28, 35, 30, 30, 31, 33, 36, 36, 34, 36, 27]
down = [2, 0, 2, 1, 3, 3, 0, 6, 7, 5, 9, 4, 11, 8, 8, 14, 15, 15, 10, 18, 19, 16, 16, 21, 20, 24, 25, 20, 27, 22, 29, 31, 26, 32, 33, 28, 33, 12]
left = [0, 1, 2, 3, 5, 5, 1, 3, 4, 9, 10, 9, 10, 14, 11, 12, 12, 16, 18, 19, 20, 22, 37, 22, 24, 25, 26, 24, 25, 27, 27, 31, 32, 33, 36, 32, 36, 19]
right = [0, 6, 2, 7, 8, 4, 6, 7, 8, 11, 12, 14, 16, 13, 13, 15, 17, 17, 37, 37, 37, 21, 21, 23, 27, 28, 28, 30, 30, 29, 30, 31, 35, 35, 34, 35, 34, 22]

for id in range(len(unique_neighborhoods)):
    zone_id.append(id)
    current_nta_name = id_to_neighborhood[id]
    current_centroid = nta_centroids[current_nta_name]
    centroid_lng.append(current_centroid.x)
    centroid_lat.append(current_centroid.y)
        
# Create the DataFrame after the loop
df_neighbor_centroid = pd.DataFrame({
        'zone_id': zone_id,
        'centroid_lng': centroid_lng,  # Assuming x is longitude
        'centroid_lat': centroid_lat,  # Assuming y is latitude
        'stay': zone_id,  # Stay in the same NTA
        'up': up,         # Up
        'down': down,     # Down
        'left': left,     # Left
        'right': right    # Right
    })
df_neighbor_centroid['zone_id'] = df_neighbor_centroid['zone_id'].astype(int)

# rl for matching
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)
# rl for matching

# rl for repositioning
def s2e(n, total_len = 14):
    n = n.astype(int)
    k = (((n[:,None] & (1 << np.arange(total_len))[::-1])) > 0).astype(np.float64)
    return k
# rl for repositioning

# rl for repositioning
def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    obtain exponential decay epsilons
    :param initial_epsilon:
    :param final_epsilon:
    :param steps:
    :param decay: decay rate
    :param pre_steps: first several epsilons does note decay
    :return:
    """
    epsilons = []

    # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)
# rl for repositioning

def distance(coord_1, coord_2):
    """
    :param coord_1: the coordinate of one point
    :type coord_1: tuple -- (latitude,longitude)
    :param coord_2: the coordinate of another point
    :type coord_2: tuple -- (latitude,longitude)
    :return: the manhattan distance between these two points
    :rtype: float
    """
    manhattan_dis = 0
    try:
        lon1,lat1 = coord_1
        lon2,lat2  = coord_2
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        r = 6371
        lat_dis = r * acos(min(1.0, cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))
        lon_dis = r * (lat2 - lat1)
        manhattan_dis = (abs(lat_dis) ** 2 + abs(lon_dis) ** 2) ** 0.5
    except Exception as e:
        print(e)
        print(coord_1)
        print(coord_2)
        print(lon1 - lon2)
        print(cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2)
        print(acos(cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))

    return manhattan_dis


def distance_array(coord_1, coord_2):
    """
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
    # manhattan_dis = list()
    # for i in range(len(coord_1)):
    #     lon1,lat1 = coord_1[i]
    #     lon2,lat2 = coord_2[i]
    #     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    #     r = 6371
    #     lat_dis = r * acos(min(1.0, cos(lat1) ** 2 * cos(lon1 - lon2) + sin(lat1) ** 2))
    #     lon_dis = r * (lat2 - lat1)
    #     manhattan_dis.append((abs(lat_dis) ** 2 + abs(lon_dis) ** 2) ** 0.5)
    # return np.array(manhattan_dis)
    coord_1 = np.array(coord_1).astype(float)
    coord_2 = np.array(coord_2).astype(float)
    coord_1_array = np.radians(coord_1)
    coord_2_array = np.radians(coord_2)
    dlon = coord_2_array[:, 0] - coord_1_array[:, 0]
    dlat = coord_2_array[:, 1] - coord_1_array[:, 1]
    a = np.sin(dlat / 2) ** 2 + np.cos(coord_1_array[:, 1]) * np.cos(coord_2_array[:, 1]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(a ** 0.5)
    r = 6371
    distance = c * r
    return distance

def get_distance_array(origin_coord_array, dest_coord_array):
    """
    :param origin_coord_array: list of coordinates
    :type origin_coord_array:  list
    :param dest_coord_array:  list of coordinates
    :type dest_coord_array:  list
    :return: tuple like (
    :rtype: list
    """
    dis_array = []
    for i in range(len(origin_coord_array)):
        dis = distance(origin_coord_array[i], dest_coord_array[i])
        dis_array.append(dis)
    dis_array = np.array(dis_array)
    return dis_array



def route_generation_array(origin_coord_array, dest_coord_array, reposition=False, mode='rg'):
    """

    :param origin_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type origin_coord_array: numpy.array
    :param dest_coord_array: the K*2 type list, the first column is lng, the second column
                                is lat.
    :type dest_coord_array: numpy.array
    :param mode: the mode of generation; if the value of mode is complete, return the last node of route;
                 if the value of mode is drop_end, the last node of route will be dropped.
    :type mode: string
    :return: tuple like (itinerary_node_list, itinerary_segment_dis_list, dis_array)
             itinerary_node_list contains the id of nodes, itinerary_segment_dis_list contains
             the distance between two nodes, dis_array contains the distance from origin node to
             destination node
    :rtype: tuple
    """
    # print("route generation start")
    # origin_coord_list为 Kx2 的array，第一列为lng，第二列为lat；dest_coord_array同理
    # itinerary_node_list的每一项为一个list，包含了对应路线中的各个节点编号
    # itinerary_segment_dis_list的每一项为一个array，包含了对应路线中的各节点到相邻下一节点的距离
    # dis_array包含了各行程的总里程
    origin_node_list = get_nodeId_from_coordinate(origin_coord_array[:, 0], origin_coord_array[:, 1])
    dest_node_list = get_nodeId_from_coordinate(dest_coord_array[:, 0], dest_coord_array[:, 1])
    itinerary_node_list = []
    itinerary_segment_dis_list = []
    dis_array = []
    if mode == 'ma':
        for origin, dest in zip(origin_node_list, dest_node_list):
            itinerary_node_list.append([dest])
            dis = distance(node_id_to_coord[origin], node_id_to_coord[dest])
            itinerary_segment_dis_list.append([dis])
            dis_array.append(dis)
        return itinerary_node_list, itinerary_segment_dis_list, np.array(dis_array)
    
    elif mode == 'rg':
        # 返回完整itinerary
        for origin, dest in zip(origin_node_list, dest_node_list):
            origin = int(origin)
            dest = int(dest)
            query = {
                'origin': origin,
                'destination': dest
            }
            re = mycollection.find_one(query)
            if re:
                ite = re['itinerary_node_list']
            else:
                ite = ox.distance.shortest_path(G, origin, dest, weight='length', cpus=16)
                if ite is None:
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
      
        for itinerary_node in itinerary_node_list:
            if itinerary_node is not None:
                itinerary_segment_dis = []
                for i in range(len(itinerary_node) - 1):
                    dis = distance(node_id_to_coord[itinerary_node[i]], node_id_to_coord[itinerary_node[i + 1]])
                    itinerary_segment_dis.append(dis)
                dis_array.append(sum(itinerary_segment_dis))
                itinerary_segment_dis_list.append(itinerary_segment_dis)
            if not reposition: 
                itinerary_node.pop()
       
    dis_array = np.array(dis_array)
    return itinerary_node_list, itinerary_segment_dis_list, dis_array

class road_network:

    def __init__(self, **kwargs):
        self.params = kwargs


    def load_data(self):
        """
        :param data_path: the path of road_network file
        :type data_path:  string
        :param file_name: the filename of road_network file
        :type file_name:  string
        :return: None
        :rtype:  None
        """
        # 路网格式：节点数字编号（从0开始），节点经度，节点纬度，所在grid id
        self.df_road_network = result


    def get_information_for_nodes(self, node_id_array):
        """
        :param node_id_array: the array of node id
        :type node_id_array:  numpy.array
        :return:  (lng_array,lat_array,grid_id_array), lng_array is the array of longitude;
                lat_array is the array of latitude; the array of node id.
        :rtype: tuple
        """
        index_list = [self.df_road_network[self.df_road_network['node_id'] == item].index[0] for item in node_id_array]
        lng_array = self.df_road_network.loc[index_list,'lng'].values
        lat_array = self.df_road_network.loc[index_list,'lat'].values
        grid_id_array = self.df_road_network.loc[index_list,'grid_id'].values
        return lng_array, lat_array, grid_id_array


def get_exponential_epsilons(initial_epsilon, final_epsilon, steps, decay=0.99, pre_steps=10):
    """
    :param initial_epsilon: initial epsilon
    :type initial_epsilon: float
    :param final_epsilon: final epsilon
    :type final_epsilon: float
    :param steps: the number of iteration
    :type steps: int
    :param decay: decay rate
    :type decay:  float
    :param pre_steps: the number of iteration of pre randomness
    :type pre_steps: int
    :return: the array of epsilon
    :rtype: numpy.array
    """

    epsilons = []

   # pre randomness
    for i in range(0, pre_steps):
        epsilons.append(deepcopy(initial_epsilon))

    # decay randomness
    epsilon = initial_epsilon
    for i in range(pre_steps, steps):
        epsilon = max(final_epsilon, epsilon * decay)
        epsilons.append(deepcopy(epsilon))

    return np.array(epsilons)


def sample_all_drivers(driver_info, t_initial, t_end, driver_sample_ratio=1, driver_number_dist=''):
    """
    :param driver_info: the information of driver
    :type driver_info:  pandas.DataFrame
    :param t_initial:   time of initial state
    :type t_initial:    int
    :param t_end:       time of terminal state
    :type t_end:        int
    :param driver_sample_ratio:
    :type driver_sample_ratio:
    :param driver_number_dist:
    :type driver_number_dist:
    :return:
    :rtype:
    """
    # 当前并无随机抽样司机；后期若需要，可设置抽样模块生成sampled_driver_info
    new_driver_info = deepcopy(driver_info)
    sampled_driver_info = new_driver_info.sample(frac=driver_sample_ratio)
    sampled_driver_info['status'] = 3
    loc_con = sampled_driver_info['start_time'] <= t_initial
    sampled_driver_info.loc[loc_con, 'status'] = 0
    sampled_driver_info['target_loc_lng'] = sampled_driver_info['lng']
    sampled_driver_info['target_loc_lat'] = sampled_driver_info['lat']
    sampled_driver_info['target_grid_id'] = sampled_driver_info['grid_id']
    sampled_driver_info['remaining_time'] = 0
    sampled_driver_info['matched_order_id'] = 'None'
    sampled_driver_info['total_idle_time'] = 0
    sampled_driver_info['time_to_last_cruising'] = 0
    sampled_driver_info['current_road_node_index'] = 0
    sampled_driver_info['remaining_time_for_current_node'] = 0
    sampled_driver_info['itinerary_node_list'] = [[] for i in range(sampled_driver_info.shape[0])]
    sampled_driver_info['itinerary_segment_time_list'] = [[] for i in range(sampled_driver_info.shape[0])]

    return sampled_driver_info


def sample_request_num(t_mean, std, delta_t):
    """
    sample request num during delta t
    :param t_mean:
    :param std:
    :param delta_t:
    :return:
    """
    random_num = np.random.normal(t_mean, std, 1)[0] * (delta_t / 100)
    random_int = random_num // 1
    random_reminder = random_num % 1

    rn = random.random()
    if rn < random_reminder:
        request_num = random_int + 1
    else:
        request_num = random_int
    return int(request_num)



def reposition(eligible_driver_table, mode):
    """
    :param eligible_driver_table:
    :type eligible_driver_table:
    :param mode:
    :type mode:
    :return:
    :rtype:
    """
    random_number = np.random.randint(0, side * side - 1)
    dest_array = []
    for _ in range(len(eligible_driver_table)):
        record = result[result['grid_id'] == random_number]
        if len(record) > 0:
            dest_array.append([record.iloc[0]['lng'], record.iloc[0]['lat']])
        else:
            dest_array.append([result.iloc[0]['lng'], result.iloc[0]['lat']])
    coord_array = eligible_driver_table.loc[:, ['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array, np.array(dest_array))
    return itinerary_node_list, itinerary_segment_dis_list, dis_array



def cruising(eligible_driver_table, mode):
    """
    :param eligible_driver_table: information of eligible driver.
    :type eligible_driver_table: pandas.DataFrame
    :param mode: the type of cruising, 'global-random' for cruising to any grid,
                 'random' for cruising to a random adjacent grid, and 'nearby'
                 for cruising to an adjacent grid or staying at the original grid.
    :type mode: string
    :param grid_adjacency: a dictionary mapping each grid_id to a list of its neighbors.
    :type grid_adjacency: dict
    :param result: a DataFrame containing lng and lat information for each grid_id.
    :type result: pandas.DataFrame
    :return: itinerary_node_list, itinerary_segment_dis_list, dis_array
    :rtype: tuple
    """
    dest_array = []
    if mode == "global-random":
        all_grid_ids = list(grid_adjacency.keys())
        dest_array = [grid_id_to_first_coords[random.choice(all_grid_ids)] for _ in eligible_driver_table['grid_id']]
    else:
        for grid_id in eligible_driver_table['grid_id']:
            # For modes 'random' and 'nearby', continue to use the loop
            if mode == 'random':
                potential_targets = grid_adjacency.get(grid_id, {grid_id})  # Use set for O(1) lookups
                random_number = random.choice(list(potential_targets))

            elif mode == 'nearby':
                potential_targets = grid_adjacency.get(grid_id, None)
                if potential_targets:
                    random_number = random.choice(list(potential_targets))
                else:
                    raise ValueError(f'Grid {grid_id} has no adjacent grids available for "nearby" mode.')

            # Fetch the lng and lat values for the chosen grid_id
            lng_lat = grid_id_to_first_coords[random_number]
            dest_array.append(lng_lat)

    # Convert dest_array to a numpy array 
    dest_array_np = np.array(dest_array)

    # Assume route_generation_array is a function you have defined elsewhere
    coord_array = eligible_driver_table[['lng', 'lat']].values
    itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(coord_array, dest_array_np)

    return itinerary_node_list, itinerary_segment_dis_list, dis_array


def skewed_normal_distribution(u,thegma,k,omega,a,input_size):

    return skewnorm.rvs(a,loc=u,scale=thegma,size=input_size)


def order_dispatch_radius(wait_requests, driver_table, dispatch_method='LD',method='pickup_distance', adjust_reward_by_radius=False):
    """
    :param wait_requests: the requests of orders
    :type wait_requests: pandas.DataFrame
    :param driver_table: the information of online drivers
    :type driver_table:  pandas.DataFrame
    :param dispatch_method: the method of order dispatch
    :type dispatch_method: string
    :return: matched_pair_actual_indexs: order and driver pair, matched_itinerary: the itinerary of matched driver
    :rtype: tuple
    """
    #  "matching_radius" is store in self.driver_table as a column
    con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
    idle_driver_table = driver_table[con_ready_to_dispatch]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]
    matched_pair_actual_indexs = []
    matched_itinerary = []
    if num_wait_request > 0 and num_idle_driver > 0:
        if dispatch_method == 'LD':
            # generate order driver pairs and corresponding itinerary
            request_array_temp = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight', 'matching_radius']]
            request_array = np.repeat(request_array_temp.values, num_idle_driver, axis=0)
            driver_loc_array_temp = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']]
            driver_loc_array = np.tile(driver_loc_array_temp.values, (num_wait_request, 1))
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
            # compare the distance with matching radius for each idle driver 
            flag = np.where(dis_array <= request_array[:, 4])[0]
            if len(flag) > 0:
                if adjust_reward_by_radius:
                    # adjust reward by radius
                    order_driver_pair = np.vstack(
                        [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3] / request_array[flag, 4], dis_array[flag]]).T
                else:
                    order_driver_pair = np.vstack(
                        [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())
                request_indexs = np.array(matched_pair_actual_indexs)[:, 0]
                driver_indexs = np.array(matched_pair_actual_indexs)[:, 1]
                request_indexs_new = []
                driver_indexs_new = []
                for index in request_indexs:
                    request_indexs_new.append(request_array_temp[request_array_temp['order_id'] == int(index)].index.tolist()[0])
                for index in driver_indexs:
                    driver_indexs_new.append(driver_loc_array_temp[driver_loc_array_temp['driver_id'] == index].index.tolist()[0])
                request_array_new = np.array(request_array_temp.loc[request_indexs_new])[:,:2]
                driver_loc_array_new = np.array(driver_loc_array_temp.loc[driver_indexs_new])[:,:2]
                itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                    driver_loc_array_new, request_array_new, mode=env_params['pickup_mode'])

                matched_itinerary = [itinerary_node_list, itinerary_segment_dis_list, dis_array]
    return matched_pair_actual_indexs, np.array(matched_itinerary)

def order_dispatch(wait_requests, driver_table, maximal_pickup_distance=1, dispatch_method='LD',method='pickup_distance'):
    """
    :param wait_requests: the requests of orders
    :type wait_requests: pandas.DataFrame
    :param driver_table: the information of online drivers
    :type driver_table:  pandas.DataFrame
    :param maximal_pickup_distance: maximum of pickup distance
    :type maximal_pickup_distance: int
    :param dispatch_method: the method of order dispatch
    :type dispatch_method: string
    :return: matched_pair_actual_indexs: order and driver pair, matched_itinerary: the itinerary of matched driver
    :rtype: tuple
    """
    con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
    idle_driver_table = driver_table[con_ready_to_dispatch]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]
    matched_pair_actual_indexs = []
    matched_itinerary = []

    if num_wait_request > 0 and num_idle_driver > 0:
        if dispatch_method == 'LD':
            # generate order driver pairs and corresponding itinerary
            request_array_temp = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']]
            request_array = np.repeat(request_array_temp.values, num_idle_driver, axis=0)
            driver_loc_array_temp = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']]
            driver_loc_array = np.tile(driver_loc_array_temp.values, (num_wait_request, 1))
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
            if method == "pickup_distance":
                # weight转换为最大pickup distance - 当前pickup distance
                request_array[:,-1] = maximal_pickup_distance - dis_array + 1
            flag = np.where(dis_array <= maximal_pickup_distance)[0]
            if len(flag) > 0:
                order_driver_pair = np.vstack(
                    [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())
                request_indexs = np.array(matched_pair_actual_indexs)[:, 0]
                driver_indexs = np.array(matched_pair_actual_indexs)[:, 1]
                request_indexs_new = []
                driver_indexs_new = []
                for index in request_indexs:
                    request_indexs_new.append(request_array_temp[request_array_temp['order_id'] == int(index)].index.tolist()[0])
                for index in driver_indexs:
                    driver_indexs_new.append(driver_loc_array_temp[driver_loc_array_temp['driver_id'] == index].index.tolist()[0])
                request_array_new = np.array(request_array_temp.loc[request_indexs_new])[:,:2]
                driver_loc_array_new = np.array(driver_loc_array_temp.loc[driver_indexs_new])[:,:2]
                itinerary_node_list, itinerary_segment_dis_list, dis_array = route_generation_array(
                    driver_loc_array_new, request_array_new, mode=env_params['pickup_mode'])

                matched_itinerary = [itinerary_node_list, itinerary_segment_dis_list, dis_array]
    return matched_pair_actual_indexs, np.array(matched_itinerary)


def driver_online_offline_decision(driver_table, current_time):

    # 注意pickup和delivery driver不应当下线
    # 车辆状态：0 cruise (park 或正在cruise)， 1 表示delivery，2 pickup, 3 表示下线, 4 reposition
    # This function is aimed to switch the driver states between 0 and 3, based on the 'start_time' and 'end_time' of drivers
    # Notice that we should not change the state of delievery and pickup drivers, since they are occopied. 
    online_driver_table = driver_table.loc[(driver_table['start_time'] <= current_time) & (driver_table['end_time'] > current_time)]
    offline_driver_table = driver_table.loc[(driver_table['start_time'] > current_time) | (driver_table['end_time'] <= current_time)]
    
    online_driver_table = online_driver_table.loc[(online_driver_table['status'] != 1) & (online_driver_table['status'] != 2)]
    offline_driver_table = offline_driver_table.loc[(offline_driver_table['status'] != 1) & (offline_driver_table['status'] != 2)]
    # print(f'online count: {len(online_driver_table)}, offline count: {len(offline_driver_table)}, total count: {len(driver_table)}')
    new_driver_table = driver_table
    new_driver_table.loc[new_driver_table.isin(online_driver_table.to_dict('list')).all(axis=1), 'status'] = 0
    new_driver_table.loc[new_driver_table.isin(offline_driver_table.to_dict('list')).all(axis=1), 'status'] = 3
    # return new_driver_table
    return new_driver_table


# define the function to get zone_id of segment node



def get_nodeId_from_coordinate(lng, lat):
    """

    :param lat: latitude
    :type lat:  float
    :param lng: longitute
    :type lng:  float
    :return:  id of node
    :rtype: string
    """
    # Using list comprehension for a more compact and slightly faster execution
    return [node_coord_to_id[(lng[i], lat[i])] for i in range(len(lat))]

def KM_simulation(wait_requests, driver_table, method = 'nothing'):
    # currently, we use the dispatch alg of peibo
    idle_driver_table = driver_table[driver_table['status'] == 0]
    num_wait_request = wait_requests.shape[0]
    num_idle_driver = idle_driver_table.shape[0]

    if num_wait_request > 0 and num_idle_driver > 0:
        starttime_1 = time.time()

        request_array = wait_requests.loc[:, ['origin_lat', 'origin_lng', 'order_id', 'weight']].values
        request_array = np.repeat(request_array, num_idle_driver, axis=0)
        driver_loc_array = idle_driver_table.loc[:, ['lat', 'lng', 'driver_id']].values
        driver_loc_array = np.tile(driver_loc_array, (num_wait_request, 1))
        assert driver_loc_array.shape[0] == request_array.shape[0]
        dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])
        # print('negative: ', np.where(dis_array)<0)
        flag = np.where(dis_array <= 950)[0]
        if method == 'pickup_distance':
            order_driver_pair = np.vstack([request_array[flag, 2], driver_loc_array[flag, 2], 951 - dis_array[flag], dis_array[flag]]).T
        elif method in ['total_travel_time_no_subway', 'total_travel_time_with_subway']:
            order_driver_pair = np.vstack(
                [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3] + 135 - dis_array[flag]/6.33, dis_array[flag]]).T
        elif method in ['sarsa_total_travel_time', 'sarsa_total_travel_time_no_subway']:
            order_driver_pair = np.vstack(
                [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3] + 135 - dis_array[flag] / 6.33,
                 dis_array[flag]]).T
        else:
            order_driver_pair = np.vstack([request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T  # rl for matching
        order_driver_pair = order_driver_pair.tolist()

        endtime_1 = time.time()
        dtime_1 = endtime_1 - starttime_1
        #print('# of pairs: ', len(order_driver_pair))
        #print("pair forming time：%.8s s" % dtime_1)

        if len(order_driver_pair) > 0:
            #matched_pair_actual_indexs = km.run_kuhn_munkres(order_driver_pair)
            matched_pair_actual_indexs = dispatch_alg_array(order_driver_pair)

            endtime_2 = time.time()
            dtime_2 = endtime_2 - endtime_1
            #print('# of matched pairs: ', len(matched_pair_actual_indexs))
            #print("dispatch alg 1 running time：%.8s s" % dtime_2)
        else:
            matched_pair_actual_indexs = []
    else:
        matched_pair_actual_indexs = []

    return matched_pair_actual_indexs

def KM_for_agent():
    # KM used in agent.py for KDD competition
    pass

def random_actions(possible_directions):
    # make random move and generate a one hot vector
    action = random.sample(possible_directions, 1)[0]
    return action

# rl for matching
# state for sarsa
class State:
    def __init__(self, time_slice: int, grid_id: int):
        self.time_slice = time_slice  # time slice
        self.grid_id = grid_id  # the grid where a taxi stays in

    def __hash__(self):
        return hash(str(self.grid_id) + str(self.time_slice))

    def __eq__(self, other):
        if self.grid_id == other.grid_id and self.time_slice == other.time_slice:
            return True
        return False
    
    def __repr__(self) -> str:
        return f"State({self.time_slice}, {self.grid_id})"
    
    def __str__(self) -> str:
        return f"({self.time_slice}, {self.grid_id})"
# rl for matching

class OrderTrajectory:
    '''
    map order_id to order trajectory - list of tuple(state, action, next_state)
    '''
    def __init__(self, gamma=0.9):
        self.trajectories = {}
        self.gamma = gamma

    def compute_discounted_rewards(self, order_id, reward):
        '''
        order_id is used as the unique identifier
        reward is the final reward recorded at the terminal state
        '''
        discounted_rewards = deque()
        trajectory = self.trajectories[order_id]
        # spread the final reward along the trajectory
        time_intervals = len(trajectory)
        Gt = reward / time_intervals
        # traverse the trajectory from back to front
        for _ in range(time_intervals):
            discounted_rewards.appendleft(Gt)
            Gt = Gt * self.gamma
        # convert deque back into list - time complexity O(n)
        return list(discounted_rewards)
    

class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def push(self, transition):
        '''
        trajectory is a list of tuples (state, action, next_state)
        discounted_rewards is a list of float 
        '''
        self.buffer.append(transition)

    def sample(self, batch_size):
        # Sample batch_size number of indices from range of buffer length
        indices = np.random.choice(range(len(self.buffer)), size=batch_size, replace=False)
        
        # Using the indices, convert the corresponding transitions to numpy arrays and split into separate arrays
        states, actions, rewards, next_states, done = zip(*np.array(self.buffer)[indices])
        
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(done)
      

    def __len__(self):
        return len(self.buffer)

#############################################################################





