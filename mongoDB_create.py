import networkx as nx
import osmnx as ox
import pymongo
from tqdm import tqdm

env_params = {
    'north_lat': 40.853214,
    'south_lat': 40.724191,
    'east_lng': -73.932459,
    'west_lng': -74.011085
} # manhattan

G = ox.graph_from_bbox(env_params['north_lat'], env_params['south_lat'], env_params['east_lng'], env_params['west_lng'],network_type='drive')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

node_id = gdf_nodes.index.tolist()
print(len(node_id))

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["route_network"]
mycollect = mydb['manhattan_smaller_road_network']
for node in tqdm(node_id):
    ite = nx.single_source_shortest_path(G, node)
    temp = []
    for key in ite.keys():
        content = {
            'node':str(node)+str(key),
            'itinerary_node_list':str(ite[key])
        }
        temp.append(content)
    mycollect.insert_many(temp)